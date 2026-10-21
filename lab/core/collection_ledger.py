"""Collection history and checks for resuming a recording."""

import json
from pathlib import Path
from dataclasses import dataclass

from lab.core.dataset_metadata import read_episode_count, read_jsonl

def read_ledger(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


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


def append_record(path: Path, record: dict) -> None:
    line = json.dumps(record) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)

#--- Session Prepare and Run ---

@dataclass(frozen=True)
class SessionRecordingState:
    episodes_before: int
    recorded_episodes: int
    remaining_episodes: int
    resume: bool


def prepare_session(
    *,
    dataset_root: Path,
    ledger_path: Path,
    plan_sha: str,
    session_index: int,
    planned_episodes: int,
) -> SessionRecordingState:
    if session_index < 1:
        raise ValueError("session_index must be positive")
    if planned_episodes < 1:
        raise ValueError("planned_episodes must be positive")

    lines = read_ledger(ledger_path)
    count, problems = check_state(
        dataset_root, lines, plan_sha
    )
    if problems:
        raise ValueError("; ".join(problems))

    done, recorded = progress(lines, session_index)
    remaining = planned_episodes - recorded

    if done or remaining <= 0:
        raise ValueError(
            f"Session {session_index} cannot be recorded again: "
            f"{recorded}/{planned_episodes} episodes recorded"
        )

    return SessionRecordingState(
        episodes_before=count or 0,
        recorded_episodes=recorded,
        remaining_episodes=remaining,
        resume=count is not None,
    )
