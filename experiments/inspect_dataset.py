"""Print what a freshly recorded LeRobotDataset actually contains.

Run right after `lerobot-record` at the robot:

    python experiments/inspect_dataset.py data/raw/pick_place_test

Checks the two things that silently break training if wrong: the joint order
in `observation.state` / `action` must equal robot.config.JOINT_ORDER, and the
camera keys must match the names in robot.config. Everything else is just
printed so you can see the shape of the data with your own eyes.

Reads the files directly (meta/info.json + parquet) so it works without
importing lerobot; handles both the v2.x and v3.x on-disk layouts.


Test without a dataset (prints this usage text):

    pip install pandas pyarrow
    python experiments/inspect_dataset.py

"""



from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    sys.exit("pandas is required: pip install pandas pyarrow")

# Runnable from anywhere without `pip install -e .`: put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot.config import CAMERAS, FPS, JOINT_ORDER  # noqa: E402

OK, BAD, INFO = "  [ok] ", "  [!!] ", "       "


def main(root: Path) -> int:
    problems = 0
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"{BAD}{info_path} not found — is this a LeRobotDataset root?")
        return 1
    info = json.loads(info_path.read_text())

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
        line = f"{INFO}{name:32s} {spec.get('dtype'):8s} shape={shape}"
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
        if names is None:
            problems += 1
            print(f"{BAD}{key}: no names recorded")
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
    found = {k.split(".")[-1] for k in features if k.startswith("observation.images.")}
    for cam in sorted(expected | found):
        if cam in expected and cam in found:
            spec = features[f"observation.images.{cam}"]
            print(f"{OK}{cam:8s} shape={spec.get('shape')}")
        elif cam in expected:
            problems += 1
            print(f"{BAD}{cam:8s} in robot.config but not in the dataset")
        else:
            problems += 1
            print(f"{BAD}{cam:8s} in the dataset but not in robot.config")

    # --- one parquet, so the columns are seen for real ---
    parquets = sorted((root / "data").rglob("*.parquet"))
    print(f"\n== parquet files: {len(parquets)}")
    if parquets:
        df = pd.read_parquet(parquets[0])
        print(f"{INFO}{parquets[0].relative_to(root)}: {len(df)} rows")
        print(f"{INFO}columns: {list(df.columns)}")
        if "episode_index" in df:
            print(f"{INFO}episodes in this file: {sorted(df['episode_index'].unique().tolist())}")
        if "timestamp" in df and len(df) > 1:
            dt = df["timestamp"].diff().dropna()
            print(f"{INFO}timestamp step: mean={dt.mean():.4f}s  max={dt.max():.4f}s  (expect ~{1 / FPS:.4f})")
            if dt.max() > 2.5 / FPS:
                problems += 1
                print(f"{BAD}frame gap > 2.5 frames — dropped frames, check camera load")
        for key in ("observation.state", "action"):
            if key in df:
                first = df[key].iloc[0]
                print(f"{INFO}{key}[0] = {[round(float(v), 2) for v in first]}")

    videos = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
    print(f"\n== video files: {len(videos)}")
    for v in videos[:4]:
        print(f"{INFO}{v.relative_to(root)}  {v.stat().st_size / 1e6:.1f} MB")

    print(f"\n{'all checks passed' if problems == 0 else f'{problems} problem(s) — see [!!] lines'}\n")
    return 0 if problems == 0 else 2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]).expanduser()))
