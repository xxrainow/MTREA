"""Read recorded joint positions for one episode."""

import json
import math
from pathlib import Path


def read_episode_joints(
    root: Path,
    episode_index: int,
) -> dict:
    import pandas as pd

    if episode_index < 0:
        raise ValueError("Episode index must be non-negative")

    info_path = root / "meta" / "info.json"
    if not info_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Metadata points outside the dataset")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info.get("features", {}).get("observation.state", {})
    joint_names = feature.get("names")

    if (
        not isinstance(joint_names, list)
        or not joint_names
        or not all(isinstance(name, str) for name in joint_names)
    ):
        raise ValueError("Joint names are missing from dataset metadata")

    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
    ]
    parts = []

    for path in sorted((root / "data").rglob("*.parquet")):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Data file points outside the dataset")

        table = pd.read_parquet(path, columns=columns)
        selected = table.loc[table["episode_index"] == episode_index]

        if not selected.empty:
            parts.append(selected)

    if not parts:
        raise FileNotFoundError(
            f"No joint data found for episode {episode_index}"
        )

    frames = pd.concat(parts, ignore_index=True)
    frames = frames.sort_values("frame_index")

    if frames["frame_index"].duplicated().any():
        raise ValueError("Duplicate frame indices in episode")

    timestamps = [float(value) for value in frames["timestamp"]]
    if (
        not all(math.isfinite(value) for value in timestamps)
        or any(b < a for a, b in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError("Invalid episode timestamps")

    positions = []
    for values in frames["observation.state"]:
        if len(values) != len(joint_names):
            raise ValueError("Joint names and position dimensions differ")

        row = [
            float(value) if math.isfinite(float(value)) else None
            for value in values
        ]
        positions.append(row)

    return {
        "episode_index": episode_index,
        "joint_names": joint_names,
        "frame_indices": [
            int(value) for value in frames["frame_index"]
        ],
        "timestamps": timestamps,
        "positions": positions,
    }
