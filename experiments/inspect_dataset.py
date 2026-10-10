"""Print what a freshly recorded LeRobotDataset actually contains.

Run right after `lerobot-record` at the robot:

    python experiments/inspect_dataset.py data/raw/pick_place_test

Checks metadata against robot.config, every state/action vector, frame counts,
and episode-local frame indices and timestamps. Does not modify the dataset.
These checks do not establish training compatibility or demonstration quality.

Reads the files directly (meta/info.json + parquet) so it works without
importing lerobot; handles both the v2.x and v3.x on-disk layouts.


Test without a dataset (prints this usage text):

    pip install pandas pyarrow
    python experiments/inspect_dataset.py

Optionally decode all MP4 files with an installed ffmpeg:

    python experiments/inspect_dataset.py data/raw/pick_place_test --check-videos
"""



from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    sys.exit("pandas is required: pip install pandas pyarrow")

# Runnable from anywhere without `pip install -e .`: put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot.config import CAMERAS, FPS, JOINT_ORDER  # noqa: E402

OK, BAD, INFO = "  [ok] ", "  [!!] ", "       "


def check_parquets(root: Path, info: dict) -> int:
    problems = 0

    def bad(message: str) -> None:
        nonlocal problems
        problems += 1
        print(f"{BAD}{message}")

    parquets = sorted((root / "data").rglob("*.parquet"))
    print(f"\n== parquet files: {len(parquets)} (checking all)")
    if not parquets:
        bad("no parquet files found under data/")
        return problems

    rows = 0
    timeline_parts = []
    required = ("observation.state", "action", "episode_index", "frame_index", "timestamp")
    for path in parquets:
        label = str(path.relative_to(root))
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            bad(f"{label}: cannot read parquet: {exc}")
            continue
        rows += len(df)
        print(f"{INFO}{label}: {len(df)} rows; columns: {list(df.columns)}")
        if df.empty:
            bad(f"{label}: empty parquet")
        missing = [key for key in required if key not in df]
        if missing:
            bad(f"{label}: missing columns {missing}")

        for key in ("observation.state", "action"):
            if key not in df:
                continue
            invalid = []
            count = 0
            for row, value in enumerate(df[key]):
                try:
                    vector = np.asarray(value)
                    valid = (vector.shape == (len(JOINT_ORDER),)
                             and vector.dtype.kind in "iuf"
                             and bool(np.isfinite(vector).all()))
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    count += 1
                    if len(invalid) < 5:
                        invalid.append(row)
            if count:
                bad(f"{label}: {key}: {count} invalid rows (sample positions {invalid}); "
                    f"expected {len(JOINT_ORDER)} finite numeric values")
            else:
                print(f"{OK}{label}: all {key} vectors valid")
            if len(df):
                print(f"{INFO}{key}[0] = {df[key].iloc[0]}")

        if all(key in df for key in required[2:]):
            timeline = df[list(required[2:])].copy()
            valid_rows = np.ones(len(df), dtype=bool)
            for key in required[2:]:
                try:
                    numeric = pd.to_numeric(timeline[key], errors="coerce").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                except (TypeError, ValueError):
                    numeric = np.full(len(df), np.nan)
                valid = np.isfinite(numeric) & (numeric >= 0)
                if key != "timestamp":
                    valid &= numeric == np.floor(numeric)
                if not valid.all():
                    bad(f"{label}: {key}: {int((~valid).sum())} invalid values")
                valid_rows &= valid
                timeline[key] = numeric
            timeline_parts.append(timeline.loc[valid_rows])

    if rows != info.get("total_frames"):
        bad(f"actual rows {rows} != meta.total_frames {info.get('total_frames')}")
    else:
        print(f"{OK}total frame count matches metadata: {rows}")

    if timeline_parts:
        timeline = pd.concat(timeline_parts, ignore_index=True)
        count = timeline["episode_index"].nunique()
        if count != info.get("total_episodes"):
            bad(f"actual episodes {count} != meta.total_episodes {info.get('total_episodes')}")
        for episode, group in timeline.groupby("episode_index", sort=True):
            # Frame order is explicit; an episode can span multiple parquet files.
            group = group.sort_values("frame_index")
            frames = group["frame_index"].to_numpy()
            if not np.array_equal(frames, np.arange(len(group))):
                bad(f"episode {int(episode)}: frame_index must start at 0 without gaps or duplicates")
            dt = group["timestamp"].diff().dropna()
            print(f"{INFO}episode {int(episode)}: {len(group)} frames")
            if len(dt):
                print(f"{INFO}timestamp step: mean={dt.mean():.4f}s max={dt.max():.4f}s "
                      f"(nominal ~{1 / FPS:.4f}s)")
                if (dt <= 0).any():
                    bad(f"episode {int(episode)}: timestamps must strictly increase")
                if (dt > 2.5 / FPS).any():
                    bad(f"episode {int(episode)}: recorded timestamp gap > 2.5 nominal frames; "
                        "this alone does not identify dropped camera frames")
    return problems


def check_videos(root: Path, features: dict, decode: bool) -> int:
    problems = 0
    videos = sorted((root / "videos").rglob("*.mp4"))
    print(f"\n== video files: {len(videos)}")
    for key, spec in features.items():
        if key.startswith("observation.images.") and spec.get("dtype") == "video":
            if not any(key in path.relative_to(root).parts for path in videos):
                problems += 1
                print(f"{BAD}{key}: no MP4 files in videos/{key}/")
    ffmpeg = shutil.which("ffmpeg") if decode else None
    if decode and ffmpeg is None:
        problems += 1
        print(f"{BAD}--check-videos requires ffmpeg; video decoding was not performed")
    for path in videos:
        label = path.relative_to(root)
        print(f"{INFO}{label}  {path.stat().st_size / 1e6:.1f} MB")
        if path.stat().st_size == 0:
            problems += 1
            print(f"{BAD}{label}: empty video file")
        elif ffmpeg:
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(path),
                 "-map", "0:v:0", "-f", "null", "-"],
                capture_output=True, text=True,
            )
            if result.returncode:
                problems += 1
                print(f"{BAD}{label}: decoding failed: {result.stderr.strip()[:500]}")
            else:
                print(f"{OK}{label}: full video decoding passed")
    if not decode:
        print(f"{INFO}video decoding NOT checked; use --check-videos with ffmpeg")
    print(f"{INFO}image content, camera identity and state/video alignment require visual review")
    return problems


def main(root: Path, check_video_decode: bool = False) -> int:
    problems = 0
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"{BAD}{info_path} not found — is this a LeRobotDataset root?")
        return 1
    try:
        info = json.loads(info_path.read_text())
        if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
            raise ValueError("expected a JSON object with a features object")
        if any(not isinstance(spec, dict) for spec in info["features"].values()):
            raise ValueError("feature specifications must be objects")
    except (OSError, ValueError) as exc:
        print(f"{BAD}cannot read metadata: {exc}")
        return 1

    print(f"\n== {root}")
    print(f"{INFO}codebase_version : {info.get('codebase_version')}")
    print(f"{INFO}robot_type       : {info.get('robot_type')}")
    print(f"{INFO}episodes / frames: {info.get('total_episodes')} / {info.get('total_frames')}")
    print(f"{INFO}fps              : {info.get('fps')}")
    if info.get("fps") != FPS:
        problems += 1
        print(f"{BAD}fps {info.get('fps')} != robot.config.FPS {FPS}")
    else:
        print(f"{OK}fps matches robot.config.FPS")

    # --- features: what each column is ---
    features = info.get("features", {})
    print("\n== features")
    for name, spec in features.items():
        shape = spec.get("shape")
        names = spec.get("names")
        line = f"{INFO}{name:32s} {str(spec.get('dtype')):8s} shape={shape}"
        if names and isinstance(names, list) and len(names) <= 8:
            line += f" names={names}"
        print(line)

    # --- joint order ---
    print("\n== joint order")
    for key in ("observation.state", "action"):
        names = features.get(key, {}).get("names")
        # LeRobot sometimes nests names as {"motors": [...]}
        if isinstance(names, dict):
            names = next(iter(names.values()), None)
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            problems += 1
            print(f"{BAD}{key}: missing or invalid names list")
            continue
        clean = [n.split(".")[0] for n in names]  # "shoulder_pan.pos" -> "shoulder_pan"
        if clean == JOINT_ORDER:
            print(f"{OK}{key} == JOINT_ORDER")
        else:
            problems += 1
            print(f"{BAD}{key} = {names}")
            print(f"{INFO}JOINT_ORDER = {JOINT_ORDER}  ← fix robot.config or the recording, not the data")

    # --- cameras ---
    print("\n== cameras")
    expected = {c.name for c in CAMERAS}
    found = {k.removeprefix("observation.images.") for k in features if k.startswith("observation.images.")}
    for cam in sorted(expected | found):
        if cam in expected and cam in found:
            spec = features[f"observation.images.{cam}"]
            print(f"{OK}{cam:8s} shape={spec.get('shape')}")
            camera = next(c for c in CAMERAS if c.name == cam)
            if spec.get("shape") != [camera.height, camera.width, 3]:
                problems += 1
                print(f"{BAD}{cam}: expected image shape {[camera.height, camera.width, 3]}")
        elif cam in expected:
            problems += 1
            print(f"{BAD}{cam:8s} in robot.config but not in the dataset")
        else:
            problems += 1
            print(f"{BAD}{cam:8s} in the dataset but not in robot.config")

    problems += check_parquets(root, info)
    problems += check_videos(root, features, check_video_decode)

    print(f"\n{'implemented checks passed' if problems == 0 else f'{problems} problem(s) — see [!!] lines'}\n")
    return 0 if problems == 0 else 2


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(__doc__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check-videos", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.root.expanduser(), args.check_videos))
