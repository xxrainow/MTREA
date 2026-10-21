"""Read recording metadata without modifying the dataset."""

import json
from pathlib import Path

# --- dataset metadata (read-only; v2.x and v3.x layouts) ---

def read_episode_count(root: Path) -> int | None:
    """total_episodes from meta/info.json, or None when there is no dataset."""
    info = root / "meta" / "info.json"
    if not info.exists():
        return None
    return int(json.loads(info.read_text(encoding="utf-8"))["total_episodes"])


def read_jsonl(path: Path) -> list[dict]:
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
        return {str(row["task"]) for row in read_jsonl(v2)}
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
        return {int(row["episode_index"]): [str(t) for t in row["tasks"]] for row in read_jsonl(v2)}
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
