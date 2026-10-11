"""Record a collection plan on the SO-101, one lerobot-record run per session.

    python experiments/collect.py --plan=research/collect/plans/pick_place_v1.yaml --dry-run
    python experiments/collect.py --plan=research/collect/plans/pick_place_v1.yaml
    python experiments/collect.py --plan=research/collect/plans/pick_place_v1.yaml --session=3

Each session prints a setup card (instruction and the per-episode layout to set
during each reset period), waits for Enter, runs lerobot-record with that
session's instruction, then checks the dataset metadata against what was asked
and appends one line to data/raw/<dataset>.collection.jsonl. That ledger is
append-only; `--session=next` (the default) picks the first session without an
ok line. Nothing here deletes or rewrites episodes.

lerobot-record behaviour this relies on, read from the v0.6.1 tag of
src/lerobot/scripts/lerobot_record.py and datasets/:
- --dataset.num_episodes is the number of episodes recorded by this run, also
  when resuming (the loop counter starts at 0), not the dataset total.
- --resume=true on a root without meta/info.json creates the directory, then
  tries to download meta/ from the Hub for repo_id, and fails. The empty
  directory it leaves makes a later create fail too (mkdir exist_ok=False).
  So resume is decided from the disk: create iff the root does not exist.
- Without --dataset.no_stamp=true, create appends a timestamp to repo_id.
- ESC ends a run early, so a session can record fewer episodes than asked; the
  ledger then says mismatch and rerunning the session records only the rest.

Ports, ids and camera indices come from robot/configs/ (as in record_test.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import yaml

# Runnable from anywhere without `pip install -e .`: put the repo root on the path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from research.collect.plan import Plan, PlanError, Session, load_plan, sessions  # noqa: E402
from robot.config import CAMERAS, FPS, ROBOT_TYPE, TELEOP_TYPE, data_root  # noqa: E402

CONFIGS = REPO_ROOT / "robot" / "configs"
LEROBOT_CHECKED = "0.6.1"


# --- dataset metadata (read-only; v2.x and v3.x layouts) ---

def read_episode_count(root: Path) -> int | None:
    """total_episodes from meta/info.json, or None when there is no dataset."""
    info = root / "meta" / "info.json"
    if not info.exists():
        return None
    return int(json.loads(info.read_text(encoding="utf-8"))["total_episodes"])


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_tasks(root: Path) -> set[str]:
    v3 = root / "meta" / "tasks.parquet"
    if v3.exists():
        import pandas as pd
        df = pd.read_parquet(v3)
        tasks = set(map(str, df.index)) if df.index.name == "task" else set()
        if "task" in df.columns:
            tasks |= set(map(str, df["task"]))
        return tasks
    v2 = root / "meta" / "tasks.jsonl"
    if v2.exists():
        return {str(row["task"]) for row in _jsonl(v2)}
    raise FileNotFoundError(f"no meta/tasks.parquet or meta/tasks.jsonl under {root}")


def read_episode_tasks(root: Path) -> dict[int, list[str]]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if files:
        import pandas as pd
        result = {}
        for path in files:
            df = pd.read_parquet(path)
            for index, tasks in zip(df["episode_index"], df["tasks"]):
                result[int(index)] = [str(t) for t in tasks]
        return result
    v2 = root / "meta" / "episodes.jsonl"
    if v2.exists():
        return {int(row["episode_index"]): [str(t) for t in row["tasks"]] for row in _jsonl(v2)}
    raise FileNotFoundError(f"no meta/episodes/*.parquet or meta/episodes.jsonl under {root}")


def verify(root: Path, before: int, requested: int, instruction: str,
           exit_code: int) -> tuple[int | None, list[str]]:
    """(episodes_after, problems) for one finished lerobot-record run."""
    problems = []
    if exit_code != 0:
        problems.append(f"lerobot-record exited with code {exit_code}")
    try:
        after = read_episode_count(root)
    except (OSError, ValueError, KeyError) as exc:
        return None, problems + [f"cannot read meta/info.json: {exc}"]
    if after is None:
        return None, problems + [f"no dataset at {root} after recording"]
    if after - before != requested:
        problems.append(f"episodes went {before} -> {after} (+{after - before}), expected +{requested}")
    try:
        if instruction not in read_tasks(root):
            problems.append(f"instruction not registered as a task: {instruction!r}")
        per_episode = read_episode_tasks(root)
        for index in range(before, after):
            if per_episode.get(index) != [instruction]:
                problems.append(f"episode {index} has tasks {per_episode.get(index)}, "
                                f"expected [{instruction!r}]")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        problems.append(f"cannot read task metadata: {exc}")
    return after, problems


# --- ledger ---

def ledger_path(plan: Plan) -> Path:
    # Beside the dataset root, not inside it: lerobot owns that directory.
    return data_root() / plan.raw_subpath.parent / f"{plan.dataset}.collection.jsonl"


def read_ledger(path: Path) -> list[dict]:
    return _jsonl(path) if path.exists() else []


def progress(lines: list[dict], index: int) -> tuple[bool, int]:
    """(done, episodes recorded so far) for one session, summed over its runs."""
    done, recorded = False, 0
    for line in lines:
        if line.get("session") != index:
            continue
        done |= line.get("status") == "ok"
        before, after = line.get("episodes_before"), line.get("episodes_after")
        if isinstance(before, int) and isinstance(after, int) and after > before:
            recorded += after - before
    return done, recorded


def pick_session(all_sessions: list[Session], lines: list[dict], arg: str) -> Session | None:
    if arg == "next":
        return next((s for s in all_sessions if not progress(lines, s.index)[0]), None)
    try:
        index = int(arg)
    except ValueError:
        raise SystemExit(f"--session must be a number or 'next', got {arg!r}")
    if not 1 <= index <= len(all_sessions):
        raise SystemExit(f"--session must be 1..{len(all_sessions)}, got {index}")
    return all_sessions[index - 1]


# --- command ---

def _cfg(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def hardware_args() -> list[str]:
    follower, leader, cams = _cfg("so101_follower.yaml"), _cfg("so101_leader.yaml"), _cfg("cameras.yaml")
    # Names, resolution and fps are fixed in robot.config; only the device index is per machine.
    cameras = {
        cam.name: {"type": "opencv", "index_or_path": cams[cam.name]["index_or_path"],
                   "width": cam.width, "height": cam.height, "fps": cam.fps}
        for cam in CAMERAS
    }
    # use_degrees sets the joint units of every recorded state/action, so the
    # yaml value is always passed rather than left to lerobot's default.
    for name, arm in (("so101_follower.yaml", follower), ("so101_leader.yaml", leader)):
        if not isinstance(arm.get("use_degrees"), bool):
            raise SystemExit(f"robot/configs/{name}: use_degrees must be true or false")
    return [
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.port={follower['port']}",
        f"--robot.id={follower['id']}",
        f"--robot.use_degrees={str(follower['use_degrees']).lower()}",
        f"--robot.cameras={json.dumps(cameras)}",
        f"--teleop.type={TELEOP_TYPE}",
        f"--teleop.port={leader['port']}",
        f"--teleop.id={leader['id']}",
        f"--teleop.use_degrees={str(leader['use_degrees']).lower()}",
    ]


def build_argv(exe: str, hardware: list[str], plan: Plan, session: Session, root: Path,
               num_episodes: int, resume: bool) -> list[str]:
    return [
        exe, *hardware,
        f"--dataset.repo_id={plan.repo_id}",
        f"--dataset.root={root}",
        f"--dataset.single_task={session.instruction}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.fps={FPS}",
        f"--dataset.episode_time_s={plan.episode_time_s}",
        f"--dataset.reset_time_s={plan.reset_time_s}",
        "--dataset.push_to_hub=false",
        "--dataset.no_stamp=true",
        f"--resume={'true' if resume else 'false'}",
    ]


def show_argv(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def setup_card(plan: Plan, session: Session, start: int, count: int, first_index: int | None) -> str:
    n = len(plan.session_objects)
    end = start + count
    lines = [
        f"==== session {session.index}/{n}: {session.object} "
        f"(episodes {start + 1}..{end} of {session.num_episodes})",
        f"instruction : {session.instruction}",
        f"target      : {plan.target} (at its fixed mark)",
        f"timing      : {plan.episode_time_s}s episode, {plan.reset_time_s}s reset",
        "",
        f"  {'ep':>3}  {'dataset#':>8}  {str(session.object) + ' at':<14}"
        + (f"  {'distractor':<12}  at" if plan.distractor else ""),
    ]
    for offset, layout in enumerate(session.schedule[start:end]):
        number = "?" if first_index is None else str(first_index + offset)
        row = f"  {start + offset + 1:>3}  {number:>8}  {layout.target_position:<14}"
        if layout.distractor is not None:
            row += f"  {str(layout.distractor):<12}  {layout.distractor_position}"
        lines.append(row)
    return "\n".join(lines)


def _git(*cmd: str) -> str | None:
    try:
        out = subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _lerobot_version() -> str | None:
    try:
        return metadata.version("lerobot")
    except metadata.PackageNotFoundError:
        return None


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def check_state(root: Path, lines: list[dict], plan_sha: str) -> tuple[int | None, list[str]]:
    """Current episode count, and reasons the dataset and ledger disagree."""
    problems = []
    if any(line.get("plan_sha256") not in (None, plan_sha) for line in lines):
        problems.append("the plan file changed after sessions were recorded with it; "
                        "a changed plan needs a new plan file and dataset name")
    count = read_episode_count(root)
    if count is None and root.exists():
        problems.append(f"{root} exists but has no meta/info.json (left by a failed run?); "
                        "lerobot-record can neither create nor resume it. Inspect it and "
                        "remove it by hand if it holds nothing")
    expected = lines[-1].get("episodes_after") if lines else 0
    if (count or 0) != expected:
        problems.append(f"dataset has {count or 0} episodes but the ledger accounts for {expected}; "
                        "episodes were recorded or lost outside this tool")
    return count, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--session", default="next", help="1-based session number, or 'next' (default)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = load_plan(args.plan)
    except PlanError as exc:
        print(f"invalid plan: {exc}", file=sys.stderr)
        return 1
    plan_sha = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    all_sessions = sessions(plan)
    root = data_root() / plan.raw_subpath
    ledger = ledger_path(plan)
    lines = read_ledger(ledger)
    count, problems = check_state(root, lines, plan_sha)
    selected = pick_session(all_sessions, lines, args.session)
    exe = shutil.which("lerobot-record")

    if args.dry_run:
        print(f"plan    : {_rel(args.plan)} (sha256 {plan_sha[:12]})")
        print(f"dataset : {root}  repo_id={plan.repo_id}  "
              f"{'missing' if count is None else f'{count} episodes'}")
        print(f"ledger  : {ledger}  ({len(lines)} lines)")
        print(f"lerobot : {_lerobot_version() or 'not installed'} "
              f"(behaviour checked against {LEROBOT_CHECKED}); "
              "--dataset.num_episodes = episodes added by this run, also with --resume")
        for problem in problems:
            print(f"[!!] {problem}")
        hardware = hardware_args()
        exists = count is not None
        index = count or 0
        for session in all_sessions:
            done, recorded = progress(lines, session.index)
            remaining = session.num_episodes - recorded
            mark = "  <-- --session=" + args.session if session is selected else ""
            print()
            if done or remaining <= 0:
                print(f"==== session {session.index}: {session.object} — recorded "
                      f"({recorded}/{session.num_episodes}){'' if done else ', but no ok line'}{mark}")
                continue
            header, rest = setup_card(plan, session, recorded, remaining, index).split("\n", 1)
            print(f"{header}{mark}\n{rest}")
            print("\n$ " + show_argv(build_argv(exe or "lerobot-record", hardware, plan, session,
                                                 root, remaining, resume=exists)))
            exists, index = True, index + remaining
        return 0

    if problems:
        for problem in problems:
            print(f"[!!] {problem}", file=sys.stderr)
        return 1
    if selected is None:
        print(f"all {len(all_sessions)} sessions have an ok line in {ledger}")
        return 0
    done, recorded = progress(lines, selected.index)
    remaining = selected.num_episodes - recorded
    if done or remaining <= 0:
        print(f"[!!] session {selected.index} already has {recorded}/{selected.num_episodes} episodes"
              f"{'' if done else ' but no ok line'}; recording it again would exceed the plan. "
              "Check the ledger by hand.", file=sys.stderr)
        return 1
    if exe is None:
        print("lerobot-record not found on PATH; activate the lerobot environment", file=sys.stderr)
        return 1
    try:
        import pandas  # noqa: F401  needed to verify v3 metadata after recording
    except ImportError:
        print("pandas and pyarrow are required: pip install pandas pyarrow", file=sys.stderr)
        return 1
    version = _lerobot_version()
    if version != LEROBOT_CHECKED:
        found = f"lerobot {version} is installed" if version else "lerobot is not importable from this Python"
        print(f"[!!] {found}; this tool's resume/num_episodes handling was checked against {LEROBOT_CHECKED}")

    before = count or 0
    resume = count is not None
    cmd = build_argv(exe, hardware_args(), plan, selected, root, remaining, resume)
    print(setup_card(plan, selected, recorded, remaining, before))
    print("\n$ " + show_argv(cmd) + "\n")
    try:
        input("Set up the scene for the first episode above, then press Enter to start (Ctrl-C aborts) ")
    except (KeyboardInterrupt, EOFError):
        print("\naborted before recording; nothing written")
        return 1

    started = _now()
    proc = subprocess.Popen(cmd)
    while True:
        try:
            exit_code = proc.wait()
            break
        except KeyboardInterrupt:
            # The child got the same Ctrl-C; let it finalize the dataset, then log what happened.
            print("\nwaiting for lerobot-record to finalize the dataset...")
    ended = _now()

    after, problems = verify(root, before, remaining, selected.instruction, exit_code)
    added = max(0, (after or before) - before)
    layouts = [
        {"episode_index": before + i, "target_position": layout.target_position,
         "distractor": None if layout.distractor is None else str(layout.distractor),
         "distractor_position": layout.distractor_position}
        for i, layout in enumerate(selected.schedule[recorded:recorded + min(added, remaining)])
    ]
    status = "ok" if not problems else "mismatch"
    record = {
        "session": selected.index,
        "object": str(selected.object),
        "instruction": selected.instruction,
        "planned_episodes": selected.num_episodes,
        "requested_episodes": remaining,
        "episodes_before": before,
        "episodes_after": after,
        "resume": resume,
        "started_at": started,
        "ended_at": ended,
        "lerobot_exit_code": exit_code,
        "lerobot_version": version,
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "plan": _rel(args.plan),
        "plan_sha256": plan_sha,
        "argv": cmd,
        "layouts": layouts,
        "status": status,
        "problems": problems,
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\nsession {selected.index}: {status}, episodes {before} -> {after}; logged to {ledger}")
    for problem in problems:
        print(f"[!!] {problem}")
    if problems:
        print("Nothing was deleted. Rerunning this session records only the episodes still missing.")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
