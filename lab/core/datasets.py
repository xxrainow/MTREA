"""Read the local recording catalog without opening hardware or changing data."""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lab.core import paths
from lab.core.dataset_metadata import read_episode_tasks, read_tasks


Backend = Literal["real", "fake"]

# Dataset names: also directory names here and on the GPU host.
DATASET_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def validate_dataset_name(name: str) -> str:
    if not isinstance(name, str) or not DATASET_NAME.fullmatch(name):
        raise ValueError("Invalid dataset name")
    return name


@dataclass
class DatasetSummary:
    id: str
    name: str
    backend: Backend
    location: str
    episodes: int | None = None
    total_frames: int | None = None
    fps: float | None = None
    robot_type: str | None = None
    cameras: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class DatasetEpisode:
    index: int
    tasks: list[str]


@dataclass
class DatasetDetail:
    dataset: DatasetSummary
    tasks: list[str]
    episodes: list[DatasetEpisode]


def _storage(base: Path, backend: Backend) -> Path:
    return paths.raw_dir(base if backend == "real" else base / "fake" / "collection")


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _summary(root: Path, base: Path, backend: Backend) -> DatasetSummary:
    summary = DatasetSummary(
        id=f"{backend}/{root.name}", name=root.name, backend=backend,
        location=str(root.relative_to(base)),
    )
    try:
        metadata = root / "meta"
        # Metadata may contain symlinks even when the dataset directory does not.
        if not _inside(metadata, root) or any(
            not _inside(path, root) for path in metadata.rglob("*")
        ):
            raise ValueError("Metadata points outside the dataset")
        info = json.loads((metadata / "info.json").read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError("meta/info.json must contain an object")
        count = info.get("total_episodes")
        if type(count) is not int or count < 0:
            raise ValueError("Invalid total_episodes in meta/info.json")
        summary.episodes = count
        frames = info.get("total_frames")
        if type(frames) is int and frames >= 0:
            summary.total_frames = frames
        fps = info.get("fps")
        if type(fps) in (int, float) and math.isfinite(fps) and fps > 0:
            summary.fps = float(fps)
        if isinstance(info.get("robot_type"), str):
            summary.robot_type = info["robot_type"]
        features = info.get("features", {})
        if isinstance(features, dict):
            summary.cameras = sorted(
                key.removeprefix("observation.images.")
                for key in features if key.startswith("observation.images.")
            )
        if backend == "real" and info.get("fake") is True:
            summary.issues.append("Fake metadata found in real storage")
    except (OSError, ValueError, TypeError) as exc:
        summary.issues.append(f"Cannot read dataset metadata: {exc}")
    return summary


def list_datasets(data_base: Path | None = None) -> list[DatasetSummary]:
    base = paths.root(data_base)
    result = []
    for backend in ("real", "fake"):
        storage = _storage(base, backend)
        if not storage.is_dir() or not _inside(storage, base):
            continue
        for root in sorted(storage.iterdir()):
            if root.is_dir() and not root.is_symlink() and (root / "meta").exists():
                result.append(_summary(root, base, backend))
    return result


def dataset_root(
    backend: Backend, name: str, data_base: Path | None = None,
) -> Path:
    base = paths.root(data_base)
    root = _storage(base, backend) / name
    if (
        name in ("", ".", "..") or "/" in name or "\\" in name
        or root.is_symlink() or not _inside(root, base)
        or not root.is_dir() or not (root / "meta").exists()
    ):
        raise FileNotFoundError("Dataset not found")
    return root


def get_dataset(
    backend: Backend, name: str, data_base: Path | None = None,
) -> DatasetDetail:
    base = paths.root(data_base)
    root = dataset_root(backend, name, data_base)
    summary = _summary(root, base, backend)
    tasks = []
    episodes = []
    if not summary.issues:
        try:
            tasks = sorted(read_tasks(root))
            episodes = [
                DatasetEpisode(index=index, tasks=instructions)
                for index, instructions in sorted(read_episode_tasks(root).items())
            ]
            if len(episodes) != summary.episodes:
                summary.issues.append("Episode metadata count differs from meta/info.json")
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            summary.issues.append(f"Cannot read episode metadata: {exc}")
    return DatasetDetail(dataset=summary, tasks=tasks, episodes=episodes)
