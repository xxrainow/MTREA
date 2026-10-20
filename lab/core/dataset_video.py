"""Resolve episode video segments from LeRobot metadata, never guessed filenames."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from lab.core.dataset_metadata import read_jsonl
from lab.core.datasets import Backend, dataset_root


@dataclass
class EpisodeVideo:
    camera: str
    start_s: float
    end_s: float | None


def _safe_file(root: Path, path: Path) -> Path:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Video metadata or file points outside the dataset")
    if not path.is_file():
        raise FileNotFoundError("Episode video or metadata file not found")
    return path


def _index(value) -> int:
    result = int(value)
    if result < 0 or result != value:
        raise ValueError("Invalid video file index")
    return result


def _resolve(
    backend: Backend, name: str, episode_index: int, data_base: Path | None,
) -> dict[str, tuple[EpisodeVideo, Path]]:
    root = dataset_root(backend, name, data_base)
    if backend == "fake":
        return {}
    info = json.loads(_safe_file(root, root / "meta/info.json").read_text())
    if not isinstance(info, dict):
        raise ValueError("Dataset info must be an object")
    features = info.get("features", {})
    if not isinstance(features, dict) or not all(isinstance(spec, dict) for spec in features.values()):
        raise ValueError("Invalid dataset features")
    keys = [key for key, spec in features.items()
            if key.startswith("observation.images.") and spec.get("dtype") == "video"]
    if not keys:
        return {}
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    row = None
    if files:
        import pandas as pd
        for file in files:
            table = pd.read_parquet(_safe_file(root, file))
            matches = table[table["episode_index"] == episode_index]
            if len(matches):
                row = matches.iloc[0].to_dict()
                break
    else:
        rows = read_jsonl(_safe_file(root, root / "meta/episodes.jsonl"))
        row = next((item for item in rows if item["episode_index"] == episode_index), None)
    if row is None:
        raise FileNotFoundError("Episode not found")

    result = {}
    for key in keys:
        camera = key.removeprefix("observation.images.")
        if files:
            prefix = f"videos/{key}"
            chunk = _index(row[f"{prefix}/chunk_index"])
            file_index = _index(row[f"{prefix}/file_index"])
            start = float(row[f"{prefix}/from_timestamp"])
            end = float(row[f"{prefix}/to_timestamp"])
        else:
            chunk = episode_index // _index(info["chunks_size"])
            file_index = episode_index
            start = 0.0
            end = float(row["length"]) / float(info["fps"]) if "length" in row else None
        if not math.isfinite(start) or start < 0 or (
            end is not None and (not math.isfinite(end) or end <= start)
        ):
            raise ValueError("Invalid episode video time range")
        relative = Path(info["video_path"].format(
            video_key=key, chunk_index=chunk, file_index=file_index,
            episode_index=episode_index, episode_chunk=chunk,
        ))
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "videos" or relative.suffix != ".mp4":
            raise ValueError("Invalid episode video path")
        # Validate existence at playback time so a missing camera does not hide others.
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Video file points outside the dataset")
        result[camera] = (EpisodeVideo(camera, start, end), path)
    return result


def episode_videos(
    backend: Backend, name: str, episode_index: int, data_base: Path | None = None,
) -> list[EpisodeVideo]:
    return [video for video, _ in _resolve(backend, name, episode_index, data_base).values()]


def episode_video_file(
    backend: Backend, name: str, episode_index: int, camera: str,
    data_base: Path | None = None,
) -> Path:
    clips = _resolve(backend, name, episode_index, data_base)
    if camera not in clips:
        raise FileNotFoundError("Camera video not found")
    root = dataset_root(backend, name, data_base)
    return _safe_file(root, clips[camera][1])
