"""What the offline check, the policy server and its check client share.

No lerobot and no torch at import time: checkpoint validation and the feature
comparison must be testable anywhere, and the fake server's refusals must match
the real server's. pyarrow and av are imported only when a dataset is read.

Verified against lerobot v0.6.1 (async_inference/helpers.py,
utils/feature_utils.py, robots/so_follower/):
- A client's lerobot_features are hw_to_dataset_features(observation_features,
  "observation", use_video=False): observation.state with names
  "<joint>.pos", and observation.images.<camera> with shape (H, W, 3).
- A checkpoint's input_features use the same keys with images as (3, H, W).
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import math
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from research.train import _git, _lerobot_version, _now, read_last_step

STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."
ACTION_KEY = "action"

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
PROCESSOR_FILES = ("policy_preprocessor.json", "policy_postprocessor.json")

# Used when serving from dataset meta alone (fake server, no checkpoint on this
# machine). PI05Config.chunk_size in v0.6.1.
DEFAULT_POLICY_TYPE = "pi05"
DEFAULT_CHUNK_SIZE = 50

# 3g.40gb reports 39.25 GiB with 0.21 GiB used on an idle slice, and H0 training
# holds 37 GiB (docs/COMPUTE.md). Expert-only training holds 13 GiB, which would
# leave room for a server, so the test is "is anyone else here", not "does it fit".
GPU_BUSY_USED_BYTES = 1 << 30
# A loaded H0 checkpoint holds 9.6 GiB of the device while serving (docs/SERVING.md).
GPU_MIN_FREE_BYTES = 12 << 30

# Stated in every report: "degrees" is true of the five body joints only.
# robots/so_follower/so_follower.py: body joints DEGREES (use_degrees=True),
# gripper RANGE_0_100.
UNITS_NOTE = "body joints in degrees; gripper in percent of its calibrated range (0-100)"


class CheckpointError(ValueError):
    """The checkpoint cannot be served. The message says why."""


class DatasetError(ValueError):
    """The dataset cannot be read for this purpose. The message says why."""


# --- host and GPU ---

def is_loopback(host: str) -> bool:
    """True only if every address `host` resolves to is a loopback address."""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(str(i[4][0]).split("%")[0]).is_loopback for i in infos)


def gpu_busy_reason(free_bytes: int, total_bytes: int) -> Optional[str]:
    """Why the GPU cannot take a server now, given torch.cuda.mem_get_info(), else None."""
    gib = 1 << 30
    used = total_bytes - free_bytes
    if used > GPU_BUSY_USED_BYTES:
        return (f"GPU busy: {used / gib:.1f} of {total_bytes / gib:.1f} GiB in use. Serving and training never "
                "share the GPU slice; stop the other process first")
    if free_bytes < GPU_MIN_FREE_BYTES:
        return f"GPU busy: only {free_bytes / gib:.1f} GiB free, serving needs {GPU_MIN_FREE_BYTES / gib:.0f} GiB"
    return None


# --- checkpoint ---

def is_hub_id(spec: str) -> bool:
    """`org/name` that is not a path on this machine."""
    parts = spec.split("/")
    return (len(parts) == 2 and all(parts) and not spec.startswith(".") and "\\" not in spec
            and ":" not in spec and not Path(spec).exists())


def _supported_policies() -> Sequence[str]:
    try:
        from lerobot.async_inference.constants import SUPPORTED_POLICIES
    except ImportError:  # validation without lerobot installed; only pi05 is ever trained here
        return (DEFAULT_POLICY_TYPE,)
    return SUPPORTED_POLICIES


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CheckpointError(f"cannot read {path} ({e.__class__.__name__}: {e})") from e


def checkpoint_provenance(checkpoint_dir: Path) -> dict:
    """Step and train run of a checkpoint inside a run directory; nulls outside one.

    The layout is <run>/lerobot/checkpoints/<step>/pretrained_model
    (docs/TRAINING_JOBS.md section 6).
    """
    resolved = checkpoint_dir.resolve()
    step_dir = resolved.parent
    checkpoints = step_dir.parent
    out = {"step": None, "run_dir": None, "run_id": None, "method": None, "seed": None, "dataset": None,
           "train_git_commit": None, "complete": None, "fake_run": False}
    if resolved.name != "pretrained_model" or not step_dir.name.isdigit():
        return out
    out["step"] = int(step_dir.name)
    if checkpoints.name != "checkpoints" or checkpoints.parent.name != "lerobot":
        return out
    last = read_last_step(checkpoints)
    out["complete"] = last is not None and out["step"] <= last
    run_dir = checkpoints.parent.parent
    config_path = run_dir / CONFIG_FILE
    if config_path.is_file():
        config = _read_json(config_path)
        args = config.get("args", {})
        out.update(run_dir=str(run_dir), run_id=run_dir.name, method=args.get("method"), seed=args.get("seed"),
                   dataset=args.get("dataset"), train_git_commit=config.get("git_commit"),
                   fake_run=bool(config.get("fake")))
    return out


def validate_checkpoint_dir(checkpoint_dir: Path) -> dict:
    """Everything a real server needs is on disk and complete. Returns the parsed config.json."""
    if not checkpoint_dir.is_dir():
        raise CheckpointError(f"{checkpoint_dir} is not a directory")
    config_path = checkpoint_dir / CONFIG_FILE
    if not config_path.is_file():
        raise CheckpointError(f"{checkpoint_dir} has no {CONFIG_FILE}")
    config = _read_json(config_path)
    policy_type = config.get("type")
    if policy_type not in _supported_policies():
        raise CheckpointError(f"policy type {policy_type!r} in {config_path} is not servable; "
                              f"supported: {list(_supported_policies())}")
    if config.get("fake"):
        raise CheckpointError(f"{checkpoint_dir} comes from a fake run and holds placeholder weights; "
                              "only --fake can serve it")
    weights = checkpoint_dir / WEIGHTS_FILE
    if not weights.is_file() or weights.stat().st_size == 0:
        raise CheckpointError(f"{checkpoint_dir} has no weights ({WEIGHTS_FILE})")
    for name in PROCESSOR_FILES:
        processor = checkpoint_dir / name
        if not processor.is_file():
            raise CheckpointError(f"{checkpoint_dir} has no processor file {name}")
        for step in _read_json(processor).get("steps", []):
            state_file = step.get("state_file")
            if state_file and not (checkpoint_dir / state_file).is_file():
                raise CheckpointError(f"{checkpoint_dir} has no {state_file}, which {name} needs")
    provenance = checkpoint_provenance(checkpoint_dir)
    if provenance["fake_run"]:
        raise CheckpointError(f"{checkpoint_dir} comes from a fake run and holds placeholder weights; "
                              "only --fake can serve it")
    if provenance["complete"] is False:
        raise CheckpointError(f"{checkpoint_dir} is not complete: lerobot's `last` has not reached step "
                              f"{provenance['step']}, so it is still being written or was cut off")
    return config


def resolve_checkpoint(spec: str) -> Path:
    """Local directory of a checkpoint path or Hub id. A Hub id is downloaded if not cached."""
    if not is_hub_id(spec):
        return Path(spec)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise CheckpointError(f"{spec} looks like a Hub id, but huggingface_hub is not installed") from e
    try:
        return Path(snapshot_download(spec, allow_patterns=[CONFIG_FILE, WEIGHTS_FILE, "policy_*"]))
    except Exception as e:  # network, auth, missing repo: all mean "cannot serve this"
        raise CheckpointError(f"cannot fetch {spec} from the Hub ({e.__class__.__name__}: {e})") from e


def _hub_config(spec: str) -> Optional[dict]:
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(spec, CONFIG_FILE)).read_text(encoding="utf-8"))
    except Exception:  # offline, no such repo, not installed: the caller reports "no config found"
        return None


def same_checkpoint(client_path: str, loaded: "ServedPolicy") -> bool:
    """The client names the loaded checkpoint: the same string, or the same directory here.

    `last` and its step directory resolve to the same place, so either is accepted.
    Nothing is loaded from this value, only compared.
    """
    if client_path == loaded.checkpoint:
        return True
    return not is_hub_id(client_path) and os.path.realpath(client_path) == loaded.resolved


@dataclass(frozen=True)
class ServedPolicy:
    """What the server accepts clients against; never derived from a client."""

    checkpoint: str  # as given on the command line; the client must send this
    resolved: str
    policy_type: str
    chunk_size: int
    input_features: dict  # key -> {"shape": [...]} as in a checkpoint's config.json
    state_names: Optional[list[str]]


def _state_names_from_config(config: dict) -> Optional[list[str]]:
    # pi05 records the dataset's action names; state and action share them on SO-101.
    return config.get("action_feature_names") or None


def served_policy_from_config(checkpoint: str, config: dict) -> ServedPolicy:
    features = config.get("input_features") or {}
    if not features:
        raise CheckpointError(f"config of {checkpoint} has no input_features")
    return ServedPolicy(
        checkpoint=checkpoint,
        resolved=os.path.realpath(checkpoint),
        policy_type=config.get("type", DEFAULT_POLICY_TYPE),
        chunk_size=int(config.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        input_features=features,
        state_names=_state_names_from_config(config),
    )


def served_policy_from_dataset(checkpoint: str, info: dict) -> ServedPolicy:
    """Fake server without a checkpoint config: what the dataset's checkpoint would expect."""
    features = {}
    for key, feature in info.get("features", {}).items():
        if key == STATE_KEY:
            features[key] = {"shape": list(feature["shape"])}
        elif key.startswith(IMAGE_PREFIX):
            height, width, channels = feature["shape"]
            features[key] = {"shape": [channels, height, width]}
    if STATE_KEY not in features or len(features) < 2:
        raise CheckpointError("dataset meta/info.json lacks observation.state or camera features")
    return ServedPolicy(checkpoint=checkpoint, resolved=os.path.realpath(checkpoint),
                        policy_type=DEFAULT_POLICY_TYPE, chunk_size=DEFAULT_CHUNK_SIZE, input_features=features,
                        state_names=info["features"][STATE_KEY].get("names"))


def fake_served_policy(checkpoint: str, dataset_root: Optional[str]) -> ServedPolicy:
    """The features a fake server validates against: checkpoint config, else dataset meta."""
    local = Path(checkpoint) / CONFIG_FILE
    config = None
    if local.is_file():
        config = _read_json(local)
    elif is_hub_id(checkpoint):
        config = _hub_config(checkpoint)
    if config is not None and config.get("input_features"):
        if config.get("type") not in _supported_policies():
            raise CheckpointError(f"policy type {config.get('type')!r} of {checkpoint} is not servable; "
                                  f"supported: {list(_supported_policies())}")
        return served_policy_from_config(checkpoint, config)
    if dataset_root is None:
        raise CheckpointError(f"no input features for {checkpoint}: it has no readable config with "
                              "input_features here; pass --dataset-root so a fake server can validate clients")
    return served_policy_from_dataset(checkpoint, read_dataset_info(Path(dataset_root)))


# --- client validation ---

def _client_image_shape(feature: dict) -> Optional[tuple[int, ...]]:
    shape = tuple(feature.get("shape", ()))
    return shape if len(shape) == 3 else None


def feature_mismatches(client: dict, served: ServedPolicy) -> list[str]:
    """Every difference between a client's lerobot_features and what the checkpoint was trained on."""
    expected = served.input_features
    out = []
    expected_cameras = {k for k in expected if k.startswith(IMAGE_PREFIX)}
    client_cameras = {k for k in client if k.startswith(IMAGE_PREFIX)}
    for key in sorted(expected_cameras - client_cameras):
        out.append(f"camera {key} is missing from the client")
    for key in sorted(client_cameras - expected_cameras):
        out.append(f"camera {key} is not in the checkpoint (it has {sorted(expected_cameras)})")
    for key in sorted(expected_cameras & client_cameras):
        channels, height, width = expected[key]["shape"]
        shape = _client_image_shape(client[key])
        if shape != (height, width, channels):
            out.append(f"camera {key} has shape {list(client[key].get('shape', ()))} (H, W, C), "
                       f"the checkpoint expects {[height, width, channels]}")
    if STATE_KEY in expected:
        state = client.get(STATE_KEY)
        if state is None:
            out.append(f"{STATE_KEY} is missing from the client")
        else:
            want_dim = expected[STATE_KEY]["shape"][0]
            shape = tuple(state.get("shape", ()))
            if shape != (want_dim,):
                out.append(f"{STATE_KEY} has shape {list(shape)}, the checkpoint expects [{want_dim}]")
            names = state.get("names")
            if served.state_names and names and list(names) != list(served.state_names):
                out.append(f"{STATE_KEY} joints are {list(names)}, the checkpoint expects "
                           f"{list(served.state_names)} in that order")
    return out


def instruction_mismatches(*, policy_type: str, pretrained_name_or_path: str, actions_per_chunk: int,
                           lerobot_features: dict, rename_map: Optional[dict], served: ServedPolicy) -> list[str]:
    """Everything wrong with a client's RemotePolicyConfig; empty means accept."""
    out = []
    if not same_checkpoint(pretrained_name_or_path, served):
        out.append(f"pretrained_name_or_path '{pretrained_name_or_path}' is not the loaded checkpoint "
                   f"'{served.checkpoint}'; this server does not load what a client asks for")
    if policy_type != served.policy_type:
        out.append(f"policy_type {policy_type!r} is not the loaded {served.policy_type!r}")
    if not 1 <= actions_per_chunk <= served.chunk_size:
        out.append(f"actions_per_chunk {actions_per_chunk} must be between 1 and the policy's "
                   f"chunk_size {served.chunk_size}")
    if rename_map:
        # The processors are loaded once, so a rename could not apply. It is also how
        # two cameras would be swapped silently.
        out.append(f"rename_map must be empty, got {rename_map}; send the checkpoint's camera names")
    out.extend(feature_mismatches(lerobot_features, served))
    return out


def lerobot_features_for(served: ServedPolicy) -> dict:
    """The lerobot_features a matching client would send. The server warms up with these."""
    features = {}
    for key, feature in served.input_features.items():
        shape = feature["shape"]
        if key == STATE_KEY:
            names = served.state_names or [f"joint_{i}.pos" for i in range(shape[0])]
            features[key] = {"dtype": "float32", "shape": (shape[0],), "names": list(names)}
        elif key.startswith(IMAGE_PREFIX):
            channels, height, width = shape
            features[key] = {"dtype": "image", "shape": (height, width, channels),
                             "names": ["height", "width", "channels"]}
    return features


# --- run records ---

def git_provenance() -> dict:
    status = _git("status", "--porcelain")
    return {"git_commit": _git("rev-parse", "HEAD") or "unknown",
            "git_dirty": None if status is None else bool(status)}


def run_provenance() -> dict:
    return {**git_provenance(), "lerobot_version": _lerobot_version(), "time": _now()}


@contextlib.contextmanager
def lerobot_log_dir(directory: Optional[Path] = None) -> Iterator[None]:
    """Import lerobot's async modules from inside this.

    PolicyServer and RobotClient call get_logger() while their classes are
    defined, which creates logs/<name>_<time>.log under the current directory.
    Without this they would litter the repo root.
    """
    target = Path(tempfile.mkdtemp(prefix="mtrea-lerobot-logs-")) if directory is None else directory
    target.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


# --- numbers ---

def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile, so a reported p95 is a latency that really happened."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))]


def latency_summary(values_ms: Sequence[float]) -> dict:
    def rounded(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, 2)

    return {"n": len(values_ms), "p50": rounded(percentile(values_ms, 50)), "p95": rounded(percentile(values_ms, 95)),
            "max": rounded(max(values_ms, default=None))}


class ChunkErrors:
    """Pools predicted-vs-recorded differences for the first 1, first 10 and all actions of a chunk."""

    HORIZONS = (1, 10, 50)

    def __init__(self, joint_names: Sequence[str]):
        self.joint_names = list(joint_names)
        n = len(self.joint_names)
        self._abs = {h: [0.0] * n for h in self.HORIZONS}
        self._sq = {h: [0.0] * n for h in self.HORIZONS}
        self._count = dict.fromkeys(self.HORIZONS, 0)

    def add(self, predicted: Sequence[Sequence[float]], recorded: Sequence[Sequence[float]]) -> None:
        """Rows are actions; the shorter of the two bounds the comparison (a chunk can outrun an episode)."""
        length = min(len(predicted), len(recorded))
        for h in self.HORIZONS:
            for t in range(min(h, length)):
                for j, (p, r) in enumerate(zip(predicted[t], recorded[t])):
                    self._abs[h][j] += abs(float(p) - float(r))
                    self._sq[h][j] += (float(p) - float(r)) ** 2
                self._count[h] += 1

    def summary(self) -> dict:
        """{"first_1": {"n", "mae": {joint: x}, "rmse": {...}, "mean_mae"}, ...}; n counts actions."""
        out = {}
        for h in self.HORIZONS:
            n = self._count[h]
            mae = {j: round(a / n, 3) if n else None for j, a in zip(self.joint_names, self._abs[h])}
            rmse = {j: round(math.sqrt(s / n), 3) if n else None for j, s in zip(self.joint_names, self._sq[h])}
            mean = round(sum(self._abs[h]) / (n * len(self.joint_names)), 3) if n else None
            out[f"first_{h}"] = {"n": n, "mae": mae, "rmse": rmse, "mean_mae": mean}
        return out


# --- dataset ---

def read_dataset_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DatasetError(f"cannot read {path} ({e.__class__.__name__}: {e})") from e


class DemoDataset:
    """Read access to a recorded LeRobot dataset, in the robot's own terms.

    Reads the parquet columns and video files directly instead of through
    LeRobotDataset: frames come back as HWC uint8, the form a robot reports, and
    replay needs no video stack. A v3.0 dataset is the only format serving reads.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = read_dataset_info(self.root)
        self.fps = int(self.info.get("fps", 0))
        features = self.info.get("features", {})
        self.cameras = sorted(k.removeprefix(IMAGE_PREFIX) for k in features if k.startswith(IMAGE_PREFIX))
        self.state_names = list(features.get(STATE_KEY, {}).get("names") or [])
        self.action_names = list(features.get(ACTION_KEY, {}).get("names") or [])
        self.joint_names = self.action_names or self.state_names
        try:
            import numpy as np
            import pyarrow.parquet as pq
        except ImportError as e:
            raise DatasetError(f"reading a dataset needs lerobot[dataset] ({e})") from e
        self._np = np
        self._pq = pq
        files = sorted((self.root / "data").glob("**/*.parquet"))
        if not files:
            raise DatasetError(f"{self.root} has no recorded frames under data/ (a fake dataset is metadata only)")
        columns = ["index", "episode_index", "frame_index", "task_index", ACTION_KEY, STATE_KEY]
        tables = [pq.read_table(f, columns=columns) for f in files]
        index = np.concatenate([t["index"].to_numpy() for t in tables])
        order = np.argsort(index)
        if not np.array_equal(index[order], np.arange(len(index))):
            raise DatasetError(f"{self.root} frame indices are not contiguous from 0")
        self.episode_index = np.concatenate([t["episode_index"].to_numpy() for t in tables])[order]
        self.frame_index = np.concatenate([t["frame_index"].to_numpy() for t in tables])[order]
        self.task_index = np.concatenate([t["task_index"].to_numpy() for t in tables])[order]
        self.actions = np.concatenate([self._vectors(t[ACTION_KEY]) for t in tables])[order]
        self.states = np.concatenate([self._vectors(t[STATE_KEY]) for t in tables])[order]
        self.episodes = sorted(int(e) for e in np.unique(self.episode_index))
        if not np.all(np.diff(self.episode_index) >= 0):
            raise DatasetError(f"{self.root} episodes are not stored in order")
        self._ranges = {e: (int(np.searchsorted(self.episode_index, e, side="left")),
                            int(np.searchsorted(self.episode_index, e, side="right"))) for e in self.episodes}
        self.tasks = self._read_tasks()
        self._video_rows: Optional[dict[int, dict]] = None

    def _vectors(self, column):
        return self._np.asarray(column.to_pylist(), dtype=self._np.float32)

    def _read_tasks(self) -> dict[int, str]:
        path = self.root / "meta" / "tasks.parquet"
        if not path.is_file():
            return {}
        table = self._pq.read_table(path)
        # v3.0 stores the task text as the pandas index, a column of its own here.
        text_col = next((c for c in table.column_names if c != "task_index"), None)
        if text_col is None:
            return {}
        return dict(zip(table["task_index"].to_pylist(), table[text_col].to_pylist()))

    def episode_range(self, episode: int) -> tuple[int, int]:
        return self._ranges[episode]

    def episode_length(self, episode: int) -> int:
        start, stop = self._ranges[episode]
        return stop - start

    def task_of(self, global_index: int) -> str:
        return self.tasks.get(int(self.task_index[global_index]), "")

    def eligible_frames(self, horizon: int) -> dict[int, range]:
        """episode -> frame indices whose next `horizon` actions stay inside the episode."""
        return {e: range(0, self.episode_length(e) - horizon + 1) for e in self.episodes
                if self.episode_length(e) >= horizon}

    def actions_from(self, episode: int, frame: int, count: int):
        """Recorded actions frame..frame+count-1 of an episode; fewer near its end."""
        start, stop = self._ranges[episode]
        return self.actions[start + frame:min(start + frame + count, stop)]

    def state_at(self, episode: int, frame: int):
        return self.states[self._ranges[episode][0] + frame]

    # --- video ---

    def _video_location(self, episode: int, camera: str) -> tuple[Path, float]:
        if self.info.get("codebase_version") != "v3.0":
            raise DatasetError(f"{self.root} is {self.info.get('codebase_version')}; serving reads v3.0 datasets")
        if self._video_rows is None:
            self._video_rows = {}
            for f in sorted((self.root / "meta" / "episodes").glob("**/*.parquet")):
                for row in self._pq.read_table(f).to_pylist():
                    self._video_rows[int(row["episode_index"])] = row
        row = self._video_rows[episode]
        key = f"{IMAGE_PREFIX}{camera}"
        path = self.root / self.info["video_path"].format(
            video_key=key, chunk_index=row[f"videos/{key}/chunk_index"], file_index=row[f"videos/{key}/file_index"])
        return path, float(row[f"videos/{key}/from_timestamp"])

    def _decode(self, path: Path, timestamps: Sequence[float]) -> list:
        """RGB frames at these increasing timestamps. Exactly one per timestamp, or an error."""
        try:
            import av
        except ImportError as e:
            raise DatasetError(f"decoding videos needs lerobot[dataset] ({e})") from e
        tolerance = 0.5 / self.fps
        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(max(0, int(timestamps[0] / stream.time_base)), stream=stream, backward=True)
            remaining = iter(timestamps)
            target = next(remaining)
            for frame in container.decode(stream):
                while target is not None and frame.time >= target - tolerance:
                    if frame.time > target + tolerance:
                        raise DatasetError(f"{path}: no frame near t={target:.3f}, next is {frame.time:.3f}")
                    frames.append(frame.to_ndarray(format="rgb24"))
                    target = next(remaining, None)
                if target is None:
                    return frames
        raise DatasetError(f"{path}: video ended before t={target:.3f}")

    def frame_images(self, episode: int, frame: int) -> dict:
        """camera -> HWC uint8 for one frame."""
        out = {}
        for camera in self.cameras:
            path, start = self._video_location(episode, camera)
            out[camera] = self._decode(path, [start + frame / self.fps])[0]
        return out

    def episode_images(self, episode: int) -> dict:
        """camera -> list of HWC uint8 for the whole episode, decoded ahead of time."""
        n = self.episode_length(episode)
        out = {}
        for camera in self.cameras:
            path, start = self._video_location(episode, camera)
            out[camera] = self._decode(path, [start + i / self.fps for i in range(n)])
        return out

    def raw_observation(self, episode: int, frame: int, images: dict, task: str) -> dict:
        """What Robot.get_observation() plus robot_client's task key would hold for this frame."""
        state = self.state_at(episode, frame)
        return {**{name: float(v) for name, v in zip(self.state_names, state)}, **images, "task": task}


def require_cameras(dataset: DemoDataset, expected: Iterable[str]) -> None:
    if sorted(expected) != dataset.cameras:
        raise DatasetError(f"{dataset.root} has cameras {dataset.cameras}, robot/config.py has {sorted(expected)}")
