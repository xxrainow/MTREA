"""Record a collection plan on the SO-101, one lerobot-record run per session.

    python experiments/collect.py --plan=research/collect/plans/pick_place_v1.yaml --dry-run
    python experiments/collect.py --plan=research/collect/plans/pick_place_v1.yaml
    python experiments/collect.py --plan=research/collect/plans/pick_place_v1.yaml --session=3

Each session prints a setup card (instruction and the per-episode layout to set
during each reset period), waits for Enter, runs lerobot-record with that
session's instruction, then checks the dataset metadata against what was asked
and appends one line to data/raw/<dataset>.collection.jsonl. That ledger is
append-only. Nothing here deletes or rewrites episodes.

--session=remaining (the default) runs every session without an ok line in plan
order, stopping at the first failure, mismatch or Ctrl+C; answering q at a
prompt stops cleanly. A rerun continues from the first session without an ok
line. --session=next runs only that session, --session=<n> only session n.

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

# Runnable from anywhere without `pip install -e .`: put the repo root on the path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from lab.core.hardware import HardwareBusy, HardwareError  # noqa: E402
from lab.core.hardware.lock import HardwareLock  # noqa: E402
from research.collect.plan import Plan, PlanError, Session, load_plan, sessions  # noqa: E402
from robot.config import data_root  # noqa: E402
from lab.core.recorder import ( RecordingProcess, build_hardware_args, build_record_command, wait_and_verify )  # noqa: E402
from lab.core.dataset_metadata import (  # noqa: E402
    read_episode_count,
)
from lab.core.collection_ledger import (  # noqa: E402
    append_record,
    check_state,
    progress,
    read_ledger,
)

LEROBOT_CHECKED = "0.6.1"


# --- ledger ---

def ledger_path(plan: Plan) -> Path:
    # Beside the dataset root, not inside it: lerobot owns that directory.
    return data_root() / plan.raw_subpath.parent / f"{plan.dataset}.collection.jsonl"



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


def hardware_args() -> list[str]:
    try:
        return build_hardware_args()
    except HardwareError as exc:
        raise SystemExit(str(exc)) from None


def build_argv(exe: str, hardware: list[str], plan: Plan, session: Session, root: Path, num_episodes: int, resume: bool) -> list[str]:
    return build_record_command(
        executable=exe,
        hardware_args=hardware,
        repo_id=plan.repo_id,
        dataset_root=root,
        instruction=session.instruction,
        num_episodes=num_episodes,
        episode_time_s=plan.episode_time_s,
        reset_time_s=plan.reset_time_s,
        resume=resume,
    )


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



def _prompt(session: Session) -> str:
    """'record', 'quit' or 'interrupted', from the operator at the keyboard."""
    while True:
        try:
            answer = input(f"Place {session.object} + distractor, then Enter to record (q to stop): ")
        except KeyboardInterrupt:
            print()
            return "interrupted"
        except EOFError:
            print()
            return "quit"
        answer = answer.strip().lower()
        if answer == "":
            return "record"
        if answer == "q":
            return "quit"
        print("press Enter to record, or q to stop")


def run_session(plan: Plan, plan_path: Path, plan_sha: str, session: Session, root: Path,
                ledger: Path, lines: list[dict], exe: str, version: str | None) -> tuple[str, str]:
    """Prompt for, record, verify and log one session.

    Returns (outcome, reason): outcome is ok, mismatch, quit, interrupted or
    error. A ledger line is appended (and added to `lines`) whenever
    lerobot-record ran.
    """
    done, recorded = progress(lines, session.index)
    remaining = session.num_episodes - recorded
    if done or remaining <= 0:
        return "error", (f"session {session.index} already has {recorded}/{session.num_episodes} "
                         f"episodes{'' if done else ' but no ok line'}; recording it again would "
                         "exceed the plan. Check the ledger by hand")
    count = read_episode_count(root)
    before = count or 0
    resume = count is not None
    cmd = build_argv(exe, hardware_args(), plan, session, root, remaining, resume)
    print()
    print(setup_card(plan, session, recorded, remaining, before))
    print("\n$ " + show_argv(cmd) + "\n")
    answer = _prompt(session)
    if answer != "record":
        return answer, "operator stopped before recording; nothing written"

    interrupted = False
    recording = RecordingProcess(cmd)

    while True:
        try:
            result = wait_and_verify(
                recording,
                dataset_root=root,
                episodes_before=before,
                requested_episodes=remaining,
                instruction=session.instruction,
            )
            break
        except KeyboardInterrupt:
            if not interrupted:
                interrupted = True
                recording.request_stop()
            print("\nwaiting for lerobot-record to finalize the dataset...")

    started = result.started_at
    ended = result.ended_at
    exit_code = result.exit_code
    after = result.episodes_after
    problems = list(result.problems)

    
    added = max(0, (after or before) - before)
    layouts = [
        {"episode_index": before + i, "target_position": layout.target_position,
         "distractor": None if layout.distractor is None else str(layout.distractor),
         "distractor_kind": layout.distractor_kind,
         "distractor_position": layout.distractor_position}
        for i, layout in enumerate(session.schedule[recorded:recorded + min(added, remaining)])
    ]
    status = "ok" if not problems else "mismatch"
    record = {
        "session": session.index,
        "object": str(session.object),
        "instruction": session.instruction,
        "planned_episodes": session.num_episodes,
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
        "plan": _rel(plan_path),
        "plan_sha256": plan_sha,
        "argv": cmd,
        "layouts": layouts,
        "status": status,
        "ok": status == "ok",
        "interrupted": interrupted,
        "problems": problems,
    }
    append_record(ledger, record)
    lines.append(record)

    print(f"\nsession {session.index}: {status}, episodes {before} -> {after}; logged to {ledger}")
    for problem in problems:
        print(f"[!!] {problem}")
    if problems:
        print("Nothing was deleted. Rerunning this session records only the episodes still missing.")
    if interrupted:
        return "interrupted", "Ctrl+C during recording"
    if problems:
        return "mismatch", "; ".join(problems)
    return "ok", ""


def print_summary(plan: Plan, root: Path, ledger: Path, done_now: list[int]) -> None:
    try:
        count = read_episode_count(root) or 0
    except (OSError, ValueError, KeyError):
        count = "unreadable"
    print("\n== summary")
    print(f"sessions ok this run : {', '.join(map(str, done_now)) or 'none'}")
    print(f"dataset episodes     : {count} / {plan.total_episodes} planned")
    print(f"ledger               : {ledger}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--session", default="remaining",
                        help="'remaining' (default): all sessions without an ok line; "
                             "'next': only the first of them; or a 1-based session number")
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
    if args.session == "remaining":
        queue = [s for s in all_sessions if not progress(lines, s.index)[0]]
        selected = None
    else:
        selected = pick_session(all_sessions, lines, args.session)
        queue = [] if selected is None else [selected]
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
    if not queue:
        print(f"all {len(all_sessions)} sessions have an ok line in {ledger}")
        print_summary(plan, root, ledger, [])
        return 0
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

    # Same lock as lab/core/hardware: no server session or CLI may open the
    # arms while lerobot-record has them.
    lock = HardwareLock()
    try:
        lock.acquire("experiments/collect.py")
    except HardwareBusy as exc:
        print(f"[!!] {exc}", file=sys.stderr)
        return 1
    try:
        done_now: list[int] = []
        for session in queue:
            outcome, reason = run_session(plan, args.plan, plan_sha, session, root, ledger, lines,
                                          exe, version)
            if outcome == "ok":
                done_now.append(session.index)
                continue
            if outcome == "quit":
                print(f"\nstopped by the operator before session {session.index} ({session.object})")
                print_summary(plan, root, ledger, done_now)
                return 0
            print(f"\n[!!] stopped at session {session.index} ({session.object}): {reason}",
                  file=sys.stderr)
            print_summary(plan, root, ledger, done_now)
            return {"interrupted": 130, "mismatch": 2}.get(outcome, 1)
        print_summary(plan, root, ledger, done_now)
        return 0
    finally:
        lock.release()

if __name__ == "__main__":
    sys.exit(main())
