"""Policy server jobs and their on-disk sessions; never opens robot hardware."""

from __future__ import annotations

import json
import os
import posixpath
import shlex
import socket
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath

from lab.core import paths, training
from lab.core.datasets import dataset_root
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.remote import Remote, RemoteError, RemoteNotConfigured
from lab.core.runner import Runner
from robot.config import FPS, POLICY_SERVER_HOST, POLICY_SERVER_PORT


def command(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def checkpoint_options(jobs: list[Job], remote: Remote | None) -> dict:
    options, errors = [], []
    for job in jobs:
        if job.spec.kind != JobKind.TRAIN or job.spec.executor != Executor.SSH or not job.spec.run_dir:
            continue
        if job.is_active():
            continue
        try:
            reader = training.run_dir_of(job, remote)
            status = training.read_json(reader, "status.json") or {}
            config = training.read_json(reader, "config.json") or {}
            if status.get("fake") or config.get("fake") or "--fake" in job.spec.cmd:
                continue
            entries = training.read_json(reader, "checkpoints.json") or {}
            for entry in entries.get("checkpoints", []):
                if entry.get("last") and entry.get("complete"):
                    relative = PurePosixPath(entry["path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        continue
                    options.append({"job_id": job.id, "step": entry["step"], "label": job.spec.label,
                                    "checkpoint": posixpath.join(job.spec.run_dir, str(relative), "pretrained_model")})
        except (RemoteError, OSError) as exc:
            errors.append(str(exc))
    return {"checkpoints": options, "errors": list(dict.fromkeys(errors))}


def start(*, target: str, mode: str, source_job_id: str | None, dataset_name: str | None,
          runner: Runner, remote: Remote | None, data_base: Path) -> Job:
    if target not in ("ssh", "local") or mode not in ("real", "hold"):
        raise ValueError("Choose GPU host or this machine, and real or fake hold mode")
    if target == "local" and mode != "hold":
        raise ValueError("This machine supports fake hold only; real serving needs the GPU host")
    session = f"serve_{uuid.uuid4().hex}"
    if target == "ssh":
        if remote is None:
            raise ValueError("GPU host not configured")
        # Validate the same configured port that the tunnel will forward.
        remote.config.policy_tunnel_argv()
        if remote.config.label != getattr(runner.executors.get(Executor.SSH), "remote", remote).label:
            raise ValueError("GPU host changed; restart the lab backend before serving")
        choices = checkpoint_options(runner.store.list(), remote)
        chosen = next((c for c in choices["checkpoints"] if c["job_id"] == source_job_id), None)
        if chosen is None:
            raise ValueError("Select a complete last checkpoint from a real GPU training run"
                             + (f": {choices['errors'][0]}" if choices["errors"] else ""))
        checkpoint = chosen["checkpoint"]
        base = remote.config.data_dir
        port = remote.config.policy_port
        session_dir = posixpath.join(base, "rollouts", "serve", session)
        argv = ["bash", "scripts/serve_policy.sh"]
    else:
        if not dataset_name:
            raise ValueError("Select a recorded dataset for fake hold's camera and joint settings")
        dataset = dataset_root("real", dataset_name, data_base)
        if not (dataset / "meta" / "info.json").is_file():
            raise ValueError("The recorded dataset has no meta/info.json")
        base = str(data_base / "fake")
        checkpoint = str(Path(base) / "serving-feature-only")
        port = POLICY_SERVER_PORT
        session_dir = str(Path(base) / "rollouts" / "serve" / session)
        argv = [sys.executable, "-m", "research.serve_policy", f"--dataset-root={dataset}"]
    # A local bind test catches an occupied endpoint before creating a remote job.
    with socket.socket() as sock:
        try:
            sock.bind((POLICY_SERVER_HOST, POLICY_SERVER_PORT))
        except OSError:
            raise ValueError(f"Local policy port {POLICY_SERVER_PORT} is already in use; stop its owner first") from None
    argv += [f"--checkpoint={checkpoint}", "--host=127.0.0.1", f"--port={port}", f"--fps={FPS}",
             f"--session={session}", f"--data-dir={base}"]
    if mode == "hold":
        argv.append("--fake=hold")
    return runner.submit(JobSpec(
        kind=JobKind.SERVE_POLICY, cmd=shlex.join(argv) if target == "ssh" else command(argv),
        executor=Executor.SSH if target == "ssh" else Executor.LOCAL,
        label=session, run_dir=session_dir, host=remote.label if target == "ssh" else None,
        metadata={"checkpoint": checkpoint, "mode": mode, "source_job_id": source_job_id,
                  "server_port": port, "client_address": f"{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}"},
    ))


def start_tunnel(job: Job, runner: Runner, remote: Remote | None) -> Job:
    if job.spec.kind != JobKind.SERVE_POLICY or job.spec.executor != Executor.SSH or not job.is_active():
        raise ValueError("A running GPU policy server is required")
    if remote is None or remote.label != job.spec.host:
        raise ValueError("The GPU host configuration no longer matches this session")
    if remote.config.policy_port != job.spec.metadata.get("server_port"):
        raise ValueError("The GPU policy port changed; stop this session and start a new one")
    return runner.submit(JobSpec(kind=JobKind.POLICY_TUNNEL,
                                cmd=command(remote.config.policy_tunnel_argv()), label=f"Tunnel · {job.spec.label}",
                                metadata={"server_job_id": job.id}))


def session_status(job: Job, jobs: list[Job], remote: Remote | None) -> dict:
    status, session, error = None, None, job.unreachable
    try:
        if job.spec.run_dir:
            if job.spec.executor == Executor.SSH:
                if remote is None or remote.label != job.spec.host:
                    raise RemoteError("GPU host configuration is missing or changed")
                names = [posixpath.join(job.spec.run_dir, n) for n in ("status.json", "session.json")]
                files = remote.transport.read_texts(names)
                texts = [files.get(n) for n in names]
            else:
                texts = []
                for name in ("status.json", "session.json"):
                    path = Path(job.spec.run_dir) / name
                    texts.append(path.read_text(encoding="utf-8") if path.is_file() else None)
            status, session = [json.loads(t) if t else None for t in texts]
    except (RemoteError, OSError, ValueError) as exc:
        error = str(exc)
    state = status.get("status") if status else "starting"
    if not job.is_active():
        if job.status == JobStatus.STOPPED:
            state = "stopped"
        elif job.status == JobStatus.FAILED:
            state = "failed"
        elif state not in ("stopped", "failed"):
            state = "ended_unknown"
    tunnels = [j for j in jobs if j.spec.kind == JobKind.POLICY_TUNNEL
               and j.spec.metadata.get("server_job_id") == job.id]
    tunnel = tunnels[0] if tunnels else None
    return {"job": job.to_dict(), "state": state, "status": status, "session": session,
            "unreachable": error, "tunnel": tunnel.to_dict() if tunnel else None,
            "client_address": job.spec.metadata.get("client_address")}
