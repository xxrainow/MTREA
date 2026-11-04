"""Policy server: one checkpoint per run, speaking LeRobot's async inference protocol.

    scripts/serve_policy.sh --checkpoint=<path or hub id> [--host=127.0.0.1] [--port=8000] [--fps=30]
        [--device=cuda] [--session=<id>] [--data-dir=<dir>] [--fake=none|hold|replay] [--dataset-root=<dir>]

The checkpoint is chosen here, never by the client. docs/SERVING.md is the
contract for the robot side; research/serve_backend.py holds what differs from
lerobot's PolicyServer.

stdout carries only parseable lines for lab's runner; everything else, lerobot's
own logging included, goes to stderr:
    STATUS loading checkpoint=<...>
    STATUS ready port=<n>
    CLIENT connected
    CLIENT disconnected        no RPC from an accepted client for CLIENT_IDLE_S
    STATUS stopped|failed exit=<n>

Every session writes <data_dir>/rollouts/serve/<session>/: session.json,
inference.jsonl, status.json, and lerobot's own log under logs/.

Exit codes: 0 never (a server runs until stopped), 1 failed while loading or
serving, 2 refused before loading, 143 SIGTERM, 130 SIGINT.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from research.serve_common import (DEFAULT_POLICY_TYPE, UNITS_NOTE, CheckpointError, DatasetError, DemoDataset,
                                   checkpoint_provenance, fake_served_policy, gpu_busy_reason, is_hub_id,
                                   is_loopback, lerobot_log_dir, resolve_checkpoint, run_provenance,
                                   served_policy_from_config, validate_checkpoint_dir)
from research.train import _now, _write_json, classify_failure

DEFAULT_PORT = 8000  # remote.yaml's policy_port; the robot side reaches it through the tunnel
CLIENT_IDLE_S = 5.0  # a live client long-polls GetActions at least every obs_queue_timeout (2 s)
GRPC_WORKERS = 4  # lerobot's serve()
STOP_GRACE_S = 1.0  # in-flight RPCs; an inference is much shorter, and a GetActions poll may be cancelled
FAKE_MODES = ("none", "hold", "replay")
SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _Stop(Exception):
    def __init__(self, signum: int):
        self.signum = signum


class _Refused(Exception):
    """Refused before anything was loaded: exit 2."""


def _say(line: str) -> None:
    print(line, flush=True)


def resolve_data_dir(arg: Optional[str], checkpoint: str) -> Path:
    """--data-dir, else ROBOLAB_DATA_ROOT, else the data dir holding the checkpoint's run, else the repo's.

    A run lives at <data_dir>/runs/<run>/..., so a real session lands beside its
    run without lab's remote.yaml being read here.
    """
    from robot.config import data_root

    if arg:
        return Path(arg)
    if os.environ.get("ROBOLAB_DATA_ROOT") or is_hub_id(checkpoint):
        return data_root()
    path = Path(os.path.realpath(checkpoint))
    for parent in path.parents:
        if parent.name == "runs":
            return parent.parent
    return data_root()


class SessionRecorder:
    """session.json, inference.jsonl and status.json. Called from gRPC worker threads."""

    def __init__(self, session_dir: Path, session: str, session_info: dict, fake: bool):
        self.dir = session_dir
        self.fake = fake
        self.session_info = {"session": session, **session_info, "fake": fake, "start_time": _now(), "end_time": None}
        self._lock = threading.Lock()
        self.dir.mkdir(parents=True)
        _write_json(self.dir / "session.json", self.session_info)
        self._log = open(self.dir / "inference.jsonl", "a", encoding="utf-8")
        self.status = {
            "status": "loading", "fake": fake, "checkpoint": session_info["checkpoint"],
            "client_connected": False, "actions_per_chunk": None,
            "counts": {"chunks": 0, "observations": 0, "observations_filtered": 0, "refusals": 0,
                       "prediction_errors": 0},
            "last_latency_ms": None, "last_refusal": None, "last_error": None, "failure": None, "port": None,
            "started_at": self.session_info["start_time"], "updated_at": None,
        }
        self._last_activity: Optional[float] = None
        self.write_status()

    def write_status(self, **changes) -> None:
        with self._lock:
            self.status.update(changes, updated_at=_now())
            _write_json(self.dir / "status.json", self.status)

    def update_session(self, **changes) -> None:
        with self._lock:
            self.session_info.update(changes)
            _write_json(self.dir / "session.json", self.session_info)

    def client_activity(self) -> None:
        self._last_activity = time.monotonic()

    def client_accepted(self, actions_per_chunk: int) -> None:
        self.client_activity()
        self.write_status(status="serving", client_connected=True, actions_per_chunk=actions_per_chunk)
        _say("CLIENT connected")

    def check_client_idle(self) -> None:
        last = self._last_activity
        if self.status["client_connected"] and last is not None and time.monotonic() - last > CLIENT_IDLE_S:
            self.write_status(status="ready", client_connected=False)
            _say("CLIENT disconnected")

    def refused(self, message: str) -> None:
        with self._lock:
            self.status["counts"]["refusals"] += 1
        self.write_status(last_refusal=message, client_connected=False,
                          **({"status": "ready"} if self.status["status"] == "serving" else {}))

    def observation(self, enqueued: bool) -> None:
        with self._lock:
            self.status["counts"]["observations"] += 1
            self.status["counts"]["observations_filtered"] += 0 if enqueued else 1

    def prediction_failed(self, message: str) -> None:
        with self._lock:
            self.status["counts"]["prediction_errors"] += 1
        self.write_status(last_error=message)

    def chunk(self, observation_t, chunk: list, timing: dict) -> None:
        import torch

        if not chunk:
            return
        actions = torch.stack([a.get_action() for a in chunk])
        timing = {k: round(v, 2) for k, v in timing.items()}
        record = {
            "fake": self.fake,
            "timestep": observation_t.get_timestep(),
            "timestamp": observation_t.get_timestamp(),
            "receive_time": getattr(observation_t, "received_at", None),
            "must_go": observation_t.must_go,
            "task": observation_t.get_observation().get("task"),
            **timing,
            "chunk_length": len(chunk),
            "chunk": [[round(float(x), 4) for x in row] for row in actions.tolist()],
            "time": _now(),
        }
        with self._lock:
            self._log.write(json.dumps(record) + "\n")
            self._log.flush()
            self.status["counts"]["chunks"] += 1
        self.write_status(last_latency_ms=timing)

    def close(self, status: str, exit_code: int, failure: Optional[str] = None) -> None:
        self.write_status(status=status, client_connected=False, failure=failure, exit_code=exit_code)
        self.update_session(end_time=_now(), exit_code=exit_code)
        with self._lock:
            self._log.close()


def _gpu_memory_gib() -> Optional[dict]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    free, total = torch.cuda.mem_get_info()
    gib = 2**30
    return {"allocated": round(torch.cuda.memory_allocated() / gib, 2),
            "reserved": round(torch.cuda.memory_reserved() / gib, 2),
            "peak_allocated": round(torch.cuda.max_memory_allocated() / gib, 2),
            "peak_reserved": round(torch.cuda.max_memory_reserved() / gib, 2),
            "device_used": round((total - free) / gib, 2), "device_total": round(total / gib, 2)}


def _check_gpu(device: str) -> None:
    import torch

    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise _Refused("CUDA is not available on this machine; use --device=cpu or --fake")
    reason = gpu_busy_reason(*torch.cuda.mem_get_info())
    if reason:
        raise _Refused(reason)


def _bind_address(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _make_grpc_server(host: str, port: int):
    """A server bound but not started: a clashing port fails before minutes of loading.

    gRPC sets SO_REUSEPORT by default on Linux, so without the option a second
    server would bind the same port silently and share the first one's clients.
    """
    from concurrent import futures

    import grpc

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=GRPC_WORKERS), options=[("grpc.so_reuseport", 0)])
    try:
        bound = server.add_insecure_port(_bind_address(host, port))
    except RuntimeError as e:
        raise _Refused(f"cannot bind {host}:{port} ({e}); is another server running?") from e
    if bound == 0:
        raise _Refused(f"cannot bind {host}:{port}; is another server running?")
    return server, bound


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts/serve_policy.sh", description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True,
                   help="pretrained_model directory or Hub id; a client must send this same string")
    p.add_argument("--host", default="127.0.0.1", help="loopback only; reach it through an SSH tunnel")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="0 picks a free port, reported in STATUS ready")
    p.add_argument("--fps", type=int, default=30, help="spacing of the action timestamps in a chunk")
    p.add_argument("--device", default="cuda", help="the client's policy_device is ignored")
    p.add_argument("--session", default=None, help="session name; must be new. Default: serve_<time>")
    p.add_argument("--data-dir", default=None,
                   help="default: ROBOLAB_DATA_ROOT, else the data dir holding the checkpoint's run, else the repo's")
    p.add_argument("--fake", choices=FAKE_MODES, default="none",
                   help="serve without a model: hold repeats the observed joints, replay returns recorded actions")
    p.add_argument("--dataset-root", default=None,
                   help="replay's actions; for a fake server without a checkpoint config, also its features")
    return p


def _preflight(args: argparse.Namespace):
    """Everything that can refuse before a session exists. Returns (served, config_dir, dataset)."""
    if not is_loopback(args.host):
        raise _Refused(f"--host {args.host!r} is not a loopback address. The protocol unpickles client data, "
                       "so the server stays on 127.0.0.1 behind an SSH tunnel")
    if args.session is not None and not SESSION_NAME.match(args.session):
        raise _Refused(f"--session {args.session!r} must be a plain name")
    fake = args.fake != "none"
    dataset = None
    if args.fake == "replay":
        if args.dataset_root is None:
            raise _Refused("--fake=replay needs --dataset-root")
        dataset = DemoDataset(Path(args.dataset_root))
        if dataset.fps != args.fps:
            raise _Refused(f"--fps {args.fps} would replay a {dataset.fps} fps dataset at the wrong speed")
    if fake:
        return fake_served_policy(args.checkpoint, args.dataset_root), None, dataset
    config_dir = resolve_checkpoint(args.checkpoint)
    config = validate_checkpoint_dir(config_dir)
    served = served_policy_from_config(args.checkpoint, config)
    _check_gpu(args.device)
    return served, config_dir, dataset


def _session_info(args: argparse.Namespace, config_dir: Optional[Path], served) -> dict:
    provenance = checkpoint_provenance(config_dir if config_dir is not None else Path(args.checkpoint))
    return {
        "checkpoint": args.checkpoint, "checkpoint_resolved": served.resolved, **provenance,
        "policy_type": served.policy_type or DEFAULT_POLICY_TYPE, "chunk_size": served.chunk_size,
        "input_features": served.input_features, "units": UNITS_NOTE,
        **run_provenance(), "python_version": platform.python_version(), "hostname": socket.gethostname(),
        "host": args.host, "device": "none" if args.fake != "none" else args.device, "fps": args.fps,
        "fake_mode": args.fake, "dataset_root": args.dataset_root, "command": sys.argv,
    }


def _serve(args: argparse.Namespace, recorder: SessionRecorder, served, config_dir, dataset, grpc_server,
           port: int) -> None:
    """Load, warm up, report ready, wait. Returns only by raising _Stop."""
    from lerobot.transport import services_pb2_grpc  # type: ignore

    with lerobot_log_dir(recorder.dir):
        from research import serve_backend

    if args.fake == "none":
        backend = serve_backend.build_backend(served, config_dir, args.device, args.fps, recorder=recorder,
                                              host=args.host, port=port)
        backend.warmup()
        recorder.update_session(gpu_memory_gib=_gpu_memory_gib())
    else:
        backend = serve_backend.build_fake_backend(served, args.fake, dataset, args.fps, recorder=recorder,
                                                   host=args.host, port=port)
    # lerobot logs a few INFO lines per observation; its file under logs/ keeps them all.
    import logging

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.WARNING)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(backend, grpc_server)
    grpc_server.start()
    recorder.write_status(status="ready", port=port)
    recorder.update_session(port=port)
    _say(f"STATUS ready port={port}")
    try:
        while True:
            time.sleep(1.0)
            recorder.check_client_idle()
    except _Stop:
        backend.stop()
        raise


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _say(f"STATUS loading checkpoint={args.checkpoint}")
    grpc_server = None
    try:
        served, config_dir, dataset = _preflight(args)
        grpc_server, port = _make_grpc_server(args.host, args.port)
        data_dir = resolve_data_dir(args.data_dir, args.checkpoint)
        session = args.session or f"serve_{datetime.now():%Y%m%d-%H%M%S}"
        session_dir = data_dir / "rollouts" / "serve" / session
        if session_dir.exists():
            raise _Refused(f"session {session_dir} already exists; past sessions are never overwritten")
    except (_Refused, CheckpointError, DatasetError, ImportError) as e:
        hint = " (the server needs lerobot[async])" if isinstance(e, ImportError) else ""
        print(f"error: {e}{hint}", file=sys.stderr, flush=True)
        if grpc_server is not None:
            grpc_server.stop(0)
        _say("STATUS failed exit=2")
        return 2
    session_dir = session_dir.resolve()
    recorder = SessionRecorder(session_dir, session, _session_info(args, config_dir, served), args.fake != "none")

    def on_signal(signum, _frame):
        raise _Stop(signum)

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    status, failure = "stopped", None
    try:
        _serve(args, recorder, served, config_dir, dataset, grpc_server, port)
        exit_code = 1  # unreachable: _serve leaves only by raising
    except _Stop as e:
        exit_code = 128 + e.signum
    except Exception as e:  # load or warmup failed: report it, as a server that never became ready
        print(f"error: {e.__class__.__name__}: {e}", file=sys.stderr, flush=True)
        status, failure, exit_code = "failed", classify_failure([str(e)]), 1
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    # A second signal during shutdown must not lose the status: the runner SIGKILLs soon after.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    grpc_server.stop(STOP_GRACE_S).wait(STOP_GRACE_S + 1.0)
    recorder.close(status, exit_code, failure)
    _say(f"STATUS {status} exit={exit_code}")
    sys.stdout.flush()
    sys.stderr.flush()
    # Skips joining gRPC's worker threads, one of which can sit in a GetActions poll, and
    # tearing down a loaded model: the files are final and there is nothing left to save.
    os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
