"""Turn a raw recording into a training-ready dataset, and prove it with checks.

    python experiments/prepare_dataset.py --raw=data/raw/pick_place_v1 \
        --out=data/datasets/pick_place_v1 \
        --plan=research/collect/plans/pick_place_v1.yaml [--check-videos]

Steps, stopping at the first failure (exit 1):
1. experiments/inspect_dataset.py on --raw must report no problems.
2. Every episode carries exactly one instruction; counts per instruction are
   printed. With --plan: each planned instruction has its planned count,
   nothing unplanned was recorded, and the protocol diversity rules hold on
   the recorded episodes.
3. pi05 normalizes state/action with QUANTILES. If --raw already has q01/q99
   in meta/stats.json it is the ready root and nothing is copied. Otherwise
   `lerobot-edit-dataset --operation.type=recompute_stats` writes a copy with
   them to --out.
4. `python -m research.train --method=h0 ... --dry-run` on the ready root must
   print `STATS ok`.
5. A summary, and the same summary as JSON in <out>.prepare.json beside --out
   (the dataset's provenance: inputs, counts, git commit, train command).

lerobot-edit-dataset recompute_stats, read from the v0.6.1 tag of
src/lerobot/scripts/lerobot_edit_dataset.py (handle_recompute_stats):
- With --new_root different from --root it copytree()s --root to --new_root
  and recomputes there; --root is not written.
- An existing --new_root is moved to <new_root>_old, replacing any earlier
  _old, so --out must not exist; and root == new_root is an in-place edit.
  Both are refused here before anything runs.
- Flags as in experiments/gpu_smoke_test.sh (--repo_id, --root, --new_root,
  --operation.type); --new_repo_id keeps the copy's repo_id equal to the raw
  one instead of lerobot's default "<repo_id>_recomputed_stats".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

# Runnable from anywhere without `pip install -e .`: repo root and this directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import read_episode_count, read_episode_tasks  # noqa: E402
from research.collect.plan import Plan, PlanError, diversity_problems, load_plan, sessions  # noqa: E402
from research.train import StatsError, load_stats, missing_quantile_stats  # noqa: E402
from robot.config import data_root  # noqa: E402

INSPECT = Path(__file__).resolve().parent / "inspect_dataset.py"
TRAIN_SEED = 0


class PrepareError(RuntimeError):
    pass


def _header(n: int, title: str) -> None:
    print(f"\n==== {n}/5 {title}", flush=True)


def _git(*cmd: str) -> str | None:
    try:
        out = subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _tail(text: str, n: int = 30) -> str:
    return "\n".join("    | " + line for line in text.splitlines()[-n:])


def _within(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or b in a.parents


def report_path(out: Path) -> Path:
    return out.parent / f"{out.name}.prepare.json"


def preflight(raw: Path, out: Path) -> None:
    if not (raw / "meta" / "info.json").is_file():
        raise PrepareError(f"--raw {raw} is not a LeRobotDataset root (no meta/info.json)")
    if out.exists():
        raise PrepareError(f"--out {out} already exists; a prepared dataset is never overwritten "
                           "(lerobot would move it to <out>_old). Pick a new --out")
    if _within(out, raw) or _within(raw, out):
        raise PrepareError(f"--out {out} and --raw {raw} overlap; the copy must live elsewhere")
    if report_path(out).exists():
        raise PrepareError(f"{report_path(out)} already exists; this dataset was prepared before")


# --- 1. integrity ---

def check_integrity(raw: Path, check_videos: bool) -> None:
    cmd = [sys.executable, str(INSPECT), str(raw), *(["--check-videos"] if check_videos else [])]
    result = subprocess.run(cmd, capture_output=True, text=True)
    problems = [line for line in result.stdout.splitlines() if line.startswith("  [!!]")]
    for line in problems:
        print(line)
    last = next((line for line in reversed(result.stdout.splitlines()) if line.strip()), "")
    if result.returncode != 0:
        raise PrepareError(f"inspect_dataset.py exited {result.returncode}: {last}\n"
                           f"    rerun it by hand for the full report: {' '.join(cmd[1:])}"
                           + (f"\n{_tail(result.stderr)}" if result.stderr.strip() else ""))
    print(f"inspect_dataset.py: {last}")


# --- 2. instructions ---

def check_instructions(raw: Path, plan: Plan | None) -> Counter:
    total = read_episode_count(raw)
    per_episode = read_episode_tasks(raw)
    indices = sorted(per_episode)
    if indices != list(range(total)):
        raise PrepareError(f"episode metadata covers {len(indices)} episodes, info.json says {total}")
    multi = {i: t for i, t in per_episode.items() if len(t) != 1}
    if multi:
        sample = dict(list(multi.items())[:5])
        raise PrepareError(f"{len(multi)} episodes do not carry exactly one instruction, e.g. {sample}")
    counts = Counter(t[0] for t in per_episode.values())
    for instruction, n in sorted(counts.items()):
        print(f"  {n:>4}  {instruction}")
    print(f"  {total:>4}  episodes total")

    if plan is None:
        print("no --plan: counts not compared with a plan")
        return counts
    planned = {s.instruction: s for s in sessions(plan)}
    problems = []
    for instruction, s in planned.items():
        if counts.get(instruction, 0) != s.num_episodes:
            problems.append(f"{s.object}: {counts.get(instruction, 0)} episodes, plan has {s.num_episodes}")
    for instruction in sorted(set(counts) - set(planned)):
        problems.append(f"unplanned instruction ({counts[instruction]} episodes): {instruction!r}")
    actual = {s.object: counts.get(instruction, 0) for instruction, s in planned.items()}
    problems += [f"recorded data: {p}" for p in diversity_problems(actual, plan.target_episodes)]
    if problems:
        raise PrepareError("dataset does not match the plan:\n" + "\n".join(f"    {p}" for p in problems))
    print(f"matches {plan.path}: every planned instruction at its planned count, "
          "nothing unplanned, protocol diversity rules hold on the recorded episodes")
    return counts


# --- 3. quantile stats ---

def _missing(repo_id: str, root: Path) -> list[str]:
    try:
        return missing_quantile_stats(load_stats(repo_id, str(root)))
    except StatsError as exc:
        raise PrepareError(str(exc)) from None


def ensure_quantiles(repo_id: str, raw: Path, out: Path) -> tuple[Path, list[str] | None]:
    """(ready root, recompute argv or None when raw already had q01/q99)."""
    missing = _missing(repo_id, raw)
    if not missing:
        print(f"{raw} already has q01/q99 for state/action; it is the training-ready root, nothing copied")
        return raw, None
    print(f"{raw} lacks q01/q99 for {missing}; recomputing into a copy at {out}")
    exe = shutil.which("lerobot-edit-dataset")
    if exe is None:
        raise PrepareError("lerobot-edit-dataset not found on PATH; activate the lerobot environment")
    cmd = [exe, f"--repo_id={repo_id}", f"--root={raw}", f"--new_repo_id={repo_id}",
           f"--new_root={out}", "--operation.type=recompute_stats"]
    print("$ " + subprocess.list2cmdline(cmd), flush=True)
    raw_stats = (raw / "meta" / "stats.json").read_bytes()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if (raw / "meta" / "stats.json").read_bytes() != raw_stats:
        raise PrepareError(f"{raw}/meta/stats.json changed during recompute_stats; --raw must stay untouched")
    if result.returncode != 0:
        raise PrepareError(f"lerobot-edit-dataset exited {result.returncode}:\n"
                           + _tail(result.stdout + result.stderr))
    still = _missing(repo_id, out)
    if still:
        raise PrepareError(f"recompute_stats finished but {out} still lacks q01/q99 for {still}")
    if read_episode_count(out) != read_episode_count(raw):
        raise PrepareError(f"{out} has {read_episode_count(out)} episodes, {raw} has {read_episode_count(raw)}")
    print(f"{out} has q01/q99 for state/action; {raw} unchanged")
    return out, cmd


# --- 4. train dry run ---

def check_train_dry_run(repo_id: str, ready: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "research.train", "--method=h0", f"--dataset={repo_id}",
               f"--dataset-root={ready}", f"--output={Path(tmp) / 'run'}", f"--seed={TRAIN_SEED}",
               "--dry-run"]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    stats_lines = [line for line in result.stdout.splitlines() if line.startswith("STATS")]
    if result.returncode != 0 or "STATS ok" not in stats_lines:
        raise PrepareError(f"research.train --dry-run exited {result.returncode}, "
                           f"STATS lines {stats_lines or 'none'}:\n" + _tail(result.stdout + result.stderr))
    print("research.train --dry-run: STATS ok")


# --- 5. summary ---

def _for_gpu_host(path: Path) -> str:
    # remote.yaml's data_dir follows the same data/ layout, so a path under the
    # local data root maps to data/<same relative path> on the GPU host.
    try:
        return "data/" + path.resolve().relative_to(data_root().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def train_command(repo_id: str, ready: Path) -> str:
    run_id = f"h0_{repo_id.split('/')[-1]}_seed{TRAIN_SEED}"
    return (f"scripts/train.sh --method=h0 --dataset={repo_id} --dataset-root={_for_gpu_host(ready)} "
            f"--output=data/runs/{run_id} --seed={TRAIN_SEED}")


def _lerobot_version() -> str | None:
    try:
        return metadata.version("lerobot")
    except metadata.PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, type=Path, help="recorded dataset root, never modified")
    parser.add_argument("--out", required=True, type=Path, help="where a recomputed copy goes; must not exist")
    parser.add_argument("--plan", type=Path, help="collection plan to check the instructions against")
    parser.add_argument("--check-videos", action="store_true", help="passed to inspect_dataset.py")
    args = parser.parse_args(argv)
    raw, out = args.raw, args.out
    try:
        plan = load_plan(args.plan) if args.plan else None
    except PlanError as exc:
        print(f"FAIL: invalid plan: {exc}", file=sys.stderr)
        return 1
    repo_id = plan.repo_id if plan else f"local/{raw.name}"

    out_existed = out.exists()
    try:
        preflight(raw, out)
        _header(1, f"integrity: inspect_dataset.py{' --check-videos' if args.check_videos else ''}")
        check_integrity(raw, args.check_videos)
        _header(2, "instructions per episode")
        counts = check_instructions(raw, plan)
        _header(3, "quantile stats (q01/q99) for pi05")
        ready, recompute_cmd = ensure_quantiles(repo_id, raw, out)
        _header(4, "research.train --dry-run")
        check_train_dry_run(repo_id, ready)
    except PrepareError as exc:
        sys.stdout.flush()  # keep the step output above the failure
        print(f"\nFAIL: {exc}", file=sys.stderr)
        if out.exists() and not out_existed:
            print(f"note: {out} was created before the failure; inspect it and remove it by hand "
                  "before rerunning", file=sys.stderr)
        return 1

    info = json.loads((ready / "meta" / "info.json").read_text(encoding="utf-8"))
    command = train_command(repo_id, ready)
    status = _git("status", "--porcelain")
    report = {
        "raw": raw.resolve().as_posix(),
        "ready": ready.resolve().as_posix(),
        "stats_recomputed": recompute_cmd is not None,
        "recompute_argv": recompute_cmd,
        "repo_id": repo_id,
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "instruction_counts": dict(sorted(counts.items())),
        "plan": args.plan.as_posix() if args.plan else None,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest() if args.plan else None,
        "check_videos": args.check_videos,
        "train_command": command,
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
        "lerobot_version": _lerobot_version(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path = report_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _header(5, "summary")
    print(f"raw      : {report['raw']}")
    print(f"ready    : {report['ready']}{' (stats recomputed)' if recompute_cmd else ' (raw, unchanged)'}")
    print(f"episodes : {report['episodes']}   frames: {report['frames']}")
    for instruction, n in report["instruction_counts"].items():
        print(f"  {n:>4}  {instruction}")
    print(f"report   : {path}")
    print(f"train on the GPU host (from repo_dir, dataset copied to the same data/ path):\n  {command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
