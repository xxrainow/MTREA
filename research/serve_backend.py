"""LeRobot's PolicyServer with the checkpoint fixed at start.

Import this inside serve_common.lerobot_log_dir(): the import creates lerobot's
log file under the current directory.

Reused from lerobot v0.6.1 unchanged: the gRPC service, chunked observation
transfer, the observation queue and similarity filter, the prediction path
(_predict_action_chunk). Changed, and why:
- SendPolicyInstructions never loads anything. It compares the client's request
  with the loaded checkpoint and aborts with FAILED_PRECONDITION on a mismatch.
  robot_client's start() catches that RpcError, logs it and returns False.
- SendObservations and GetActions refuse a client that was not accepted.
- GetActions answers a timeout with an empty Actions. Upstream returns Empty,
  which the Actions serializer cannot encode, so the client would log an RPC
  error every obs_queue_timeout whenever the similarity filter holds back
  observations. Its receive loop already skips an Actions with no data.
- Ready also forgets the previous client's last observation. Upstream keeps it,
  which would compare a new client's first observations with an old session.
"""

from __future__ import annotations

import contextlib
import io
import pickle  # nosec: the protocol itself; the host is loopback only (serve_common.is_loopback)
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.transport import services_pb2  # type: ignore

from research.serve_common import STATE_KEY, ServedPolicy, instruction_mismatches, lerobot_features_for

# Printed by PI05Policy.from_pretrained. On any load failure, a strict key mismatch
# included, it prints a warning and returns a randomly initialized model, so this
# line is the only evidence the weights arrived.
PI05_WEIGHTS_LOADED = "All keys loaded successfully!"


class WeightsNotLoaded(RuntimeError):
    pass


class _Tee(io.TextIOBase):
    """Sends text to stderr, keeping stdout for the parseable lines, and remembers it."""

    def __init__(self):
        self.text = []

    def write(self, s: str) -> int:
        self.text.append(s)
        return sys.stderr.write(s)


def load_policy(config_dir: Path, device: str, weights: Optional[str] = None):
    """Policy and processors from a checkpoint, as PolicyServer.SendPolicyInstructions loads them.

    `weights` runs other weights behind this checkpoint's config and processors.
    """
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(str(config_dir))
    config.device = device
    captured = _Tee()
    with contextlib.redirect_stdout(captured):
        policy = get_policy_class(config.type).from_pretrained(weights or str(config_dir), config=config)
    if config.type == "pi05" and PI05_WEIGHTS_LOADED not in "".join(captured.text):
        raise WeightsNotLoaded(f"weights of {weights or config_dir} did not load: lerobot fell back to a "
                               f"randomly initialized model. Its output:\n{''.join(captured.text)}")
    policy.to(device)
    device_override = {"device": device}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(config_dir),
        preprocessor_overrides={"device_processor": device_override,
                                "rename_observations_processor": {"rename_map": {}}},
        postprocessor_overrides={"device_processor": device_override},
    )
    return policy, preprocessor, postprocessor


class _TimedCall:
    """Times one callable lerobot's prediction path calls, without changing that path."""

    def __init__(self, fn: Callable, marks: dict, name: str):
        self.fn, self.marks, self.name = fn, marks, name

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.fn(*args, **kwargs)
        self.marks[self.name] = (start, time.perf_counter())
        return result


class PolicyBackend(PolicyServer):
    """One checkpoint, chosen before any client connects. See the module docstring."""

    def __init__(self, config: PolicyServerConfig, served: ServedPolicy, recorder=None):
        super().__init__(config)
        self.served = served
        self.recorder = recorder
        self.accepted = False
        self.last_timing: dict = {}
        self._marks: dict = {}
        self.lerobot_features = lerobot_features_for(served)
        self.actions_per_chunk = served.chunk_size
        self.policy_type = served.policy_type

    def attach(self, policy, preprocessor, postprocessor, device: str) -> None:
        self.policy = policy
        self.device = device
        self.preprocessor = _TimedCall(preprocessor, self._marks, "preprocess")
        self.postprocessor = postprocessor

    def warmup(self) -> None:
        """One inference on a dummy observation, so the first client does not pay for lazy setup."""
        raw = {"task": "warmup"}
        for key, feature in self.lerobot_features.items():
            if key == STATE_KEY:
                raw.update(dict.fromkeys(feature["names"], 0.0))
            else:
                import numpy as np

                raw[key.rsplit(".", 1)[-1]] = np.zeros(feature["shape"], dtype=np.uint8)
        self._predict_action_chunk(TimedObservation(time.time(), 0, raw))
        self.last_processed_obs = None

    # --- gRPC ---

    def _touch(self) -> None:
        if self.recorder is not None:
            self.recorder.client_activity()

    def _refuse(self, context, message: str) -> None:
        print(f"REFUSED {message}", file=sys.stderr, flush=True)
        if self.recorder is not None:
            self.recorder.refused(message)
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)

    def Ready(self, request, context):  # noqa: N802
        self.accepted = False
        self.last_processed_obs = None
        self.fps_tracker.reset()
        self._touch()
        return super().Ready(request, context)

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        self.accepted = False
        try:
            specs = pickle.loads(request.data)  # nosec
        except Exception as e:  # anything a malformed pickle can raise
            self._refuse(context, f"policy instructions cannot be decoded ({e.__class__.__name__}: {e})")
        if not isinstance(specs, RemotePolicyConfig):
            self._refuse(context, f"policy instructions must be a RemotePolicyConfig, got {type(specs).__name__}")
        problems = instruction_mismatches(
            policy_type=specs.policy_type, pretrained_name_or_path=specs.pretrained_name_or_path,
            actions_per_chunk=specs.actions_per_chunk, lerobot_features=specs.lerobot_features,
            rename_map=specs.rename_map, served=self.served)
        if problems:
            self._refuse(context, "refused: " + "; ".join(problems))
        self.lerobot_features = specs.lerobot_features
        self.actions_per_chunk = specs.actions_per_chunk
        self.accepted = True
        self.logger.info(f"Accepted {context.peer()}: {specs.actions_per_chunk} actions per chunk "
                         f"of {self.served.checkpoint} (client device {specs.device!r} ignored)")
        if self.recorder is not None:
            self.recorder.client_accepted(specs.actions_per_chunk)
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        self._touch()
        if not self.accepted:
            self._refuse(context, "observation before accepted policy instructions")
        return super().SendObservations(request_iterator, context)

    def GetActions(self, request, context):  # noqa: N802
        self._touch()
        if not self.accepted:
            self._refuse(context, "GetActions before accepted policy instructions")
        result = super().GetActions(request, context)
        return result if isinstance(result, services_pb2.Actions) else services_pb2.Actions()

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        obs.received_at = time.time()
        enqueued = super()._enqueue_observation(obs)
        if self.recorder is not None:
            self.recorder.observation(enqueued)
        return enqueued

    # --- prediction ---

    def _get_action_chunk(self, observation):
        start = time.perf_counter()
        chunk = super()._get_action_chunk(observation)
        if chunk.is_cuda:
            torch.cuda.synchronize()  # else queued GPU work is billed to postprocess
        self._marks["inference"] = (start, time.perf_counter())
        return chunk

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        start = time.perf_counter()
        try:
            chunk = super()._predict_action_chunk(observation_t)
        except Exception as e:
            # Upstream's GetActions swallows this and returns nothing, so record it here.
            print(f"error: prediction failed: {e.__class__.__name__}: {e}", file=sys.stderr, flush=True)
            if self.recorder is not None:
                self.recorder.prediction_failed(f"{e.__class__.__name__}: {e}")
            raise
        end = time.perf_counter()
        pre_start, pre_end = self._marks["preprocess"]
        inf_start, inf_end = self._marks["inference"]
        self.last_timing = {
            "prepare_ms": (pre_start - start) * 1000, "preprocess_ms": (pre_end - pre_start) * 1000,
            "inference_ms": (inf_end - inf_start) * 1000, "postprocess_ms": (end - inf_end) * 1000,
            "total_ms": (end - start) * 1000,
        }
        self._record(observation_t, chunk)
        return chunk

    def _record(self, observation_t: TimedObservation, chunk: list[TimedAction]) -> None:
        if self.recorder is not None:
            self.recorder.chunk(observation_t, chunk, self.last_timing)


def build_backend(served: ServedPolicy, config_dir: Path, device: str, fps: int, *, weights: Optional[str] = None,
                  recorder=None, host: str = "127.0.0.1", port: int = 8000) -> PolicyBackend:
    """A real backend, policy loaded and on its device. Warmup is the caller's call."""
    # port 0 means "pick one" to gRPC; PolicyServerConfig's own range check starts at 1.
    config = PolicyServerConfig(host=host, port=port or 1, fps=fps)
    backend = PolicyBackend(config, served, recorder)
    backend.attach(*load_policy(config_dir, device, weights), device)
    return backend
