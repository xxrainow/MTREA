"""experiments/collect.py session loop, against a fake lerobot-record; no robot."""

import importlib.util
import io
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "research/collect/plans/pick_place_v1.yaml"

spec = importlib.util.spec_from_file_location("collect", REPO_ROOT / "experiments/collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)

# Writes what lerobot v0.6.1 lerobot-record leaves on disk (v3 metadata only),
# with its create/resume rules. When the instruction contains FAKE_SHORT_TASK
# it records one episode too few; FAKE_FAIL_TASK, it exits 1 having recorded none.
FAKE = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    import pandas as pd

    args = dict(a[2:].split("=", 1) for a in sys.argv[1:])
    json.loads(args["robot.cameras"])
    assert args["dataset.no_stamp"] == "true"
    assert args["robot.use_degrees"] in ("true", "false")
    root, task = Path(args["dataset.root"]), args["dataset.single_task"]
    n = int(args["dataset.num_episodes"])
    if os.environ.get("FAKE_FAIL_TASK") and os.environ["FAKE_FAIL_TASK"] in task:
        sys.exit(1)
    if os.environ.get("FAKE_SHORT_TASK") and os.environ["FAKE_SHORT_TASK"] in task:
        n -= 1
    meta = root / "meta"
    episodes_path = meta / "episodes/chunk-000/file-000.parquet"
    if args["resume"] == "true":
        root.mkdir(parents=True, exist_ok=True)
        if not (meta / "info.json").exists():
            sys.exit("RepositoryNotFoundError")
        info = json.loads((meta / "info.json").read_text())
        tasks = pd.read_parquet(meta / "tasks.parquet")
        episodes = pd.read_parquet(episodes_path)
    else:
        root.mkdir(parents=True, exist_ok=False)
        info = {"codebase_version": "v3.0", "total_episodes": 0, "fps": 30}
        tasks = pd.DataFrame({"task_index": []}, index=pd.Index([], name="task"))
        episodes = pd.DataFrame({"episode_index": [], "tasks": []})
    if task not in tasks.index:
        new = pd.DataFrame({"task_index": [len(tasks)]}, index=pd.Index([task], name="task"))
        tasks = pd.concat([tasks, new])
    start = info["total_episodes"]
    new = pd.DataFrame({"episode_index": list(range(start, start + n)), "tasks": [[task]] * n})
    episodes = pd.concat([episodes, new], ignore_index=True)
    info["total_episodes"] += n
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info))
    tasks.to_parquet(meta / "tasks.parquet")
    episodes.to_parquet(episodes_path)
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = tmp_path / "fake_record.py"
    fake.write_text(FAKE, encoding="utf-8")
    if os.name == "nt":
        (bin_dir / "lerobot-record.cmd").write_text(f'@"{sys.executable}" "{fake}" %*\r\n')
    else:
        exe = bin_dir / "lerobot-record"
        exe.write_text(f"#!{sys.executable}\n" + FAKE, encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ROBOLAB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("FAKE_SHORT_TASK", raising=False)
    monkeypatch.delenv("FAKE_FAIL_TASK", raising=False)
    return tmp_path / "data"


def run(monkeypatch, keys: str, *args: str) -> int:
    monkeypatch.setattr("sys.stdin", io.StringIO(keys))
    return collect.main([f"--plan={PLAN}", *args])


def ledger(data: Path) -> list[dict]:
    path = data / "raw" / "pick_place_v1.collection.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def episodes(data: Path) -> int:
    return collect.read_episode_count(data / "raw" / "pick_place_v1")


def test_records_all_sessions_in_one_run(env, monkeypatch, capsys):
    assert run(monkeypatch, "\n" * 6) == 0
    assert episodes(env) == 50
    lines = ledger(env)
    assert [line["session"] for line in lines] == [1, 2, 3, 4, 5, 6]
    assert all(line["ok"] and line["status"] == "ok" for line in lines)
    assert [line["resume"] for line in lines] == [False] + [True] * 5
    assert all(layout["distractor_kind"] for line in lines for layout in line["layouts"])
    out = capsys.readouterr().out
    assert "sessions ok this run : 1, 2, 3, 4, 5, 6" in out
    assert "50 / 50 planned" in out
    # Nothing left: a rerun records nothing and still exits 0.
    assert run(monkeypatch, "") == 0
    assert len(ledger(env)) == 6


def test_q_stops_cleanly_before_a_session(env, monkeypatch, capsys):
    assert run(monkeypatch, "\n\nq\n") == 0
    assert [line["session"] for line in ledger(env)] == [1, 2]
    assert episodes(env) == 17
    out = capsys.readouterr().out
    assert "stopped by the operator before session 3" in out
    assert "17 / 50 planned" in out


def test_mismatch_stops_the_loop_and_a_rerun_resumes_there(env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SHORT_TASK", "green block")  # session 3
    assert run(monkeypatch, "\n" * 6) == 2
    lines = ledger(env)
    assert [line["session"] for line in lines] == [1, 2, 3]
    assert lines[-1]["ok"] is False and lines[-1]["status"] == "mismatch"
    assert episodes(env) == 9 + 8 + 7
    assert "stopped at session 3" in capsys.readouterr().err

    monkeypatch.delenv("FAKE_SHORT_TASK")
    assert run(monkeypatch, "\n" * 6) == 0
    resumed = ledger(env)[3:]
    assert [line["session"] for line in resumed] == [3, 4, 5, 6]
    assert resumed[0]["requested_episodes"] == 1  # only the missing episode
    assert resumed[0]["layouts"][0]["episode_index"] == 24
    assert episodes(env) == 50


def test_lerobot_failure_stops_the_loop(env, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_FAIL_TASK", "blue block")  # session 2
    assert run(monkeypatch, "\n" * 6) != 0
    lines = ledger(env)
    assert [line["session"] for line in lines] == [1, 2]
    assert lines[-1]["lerobot_exit_code"] == 1 and lines[-1]["ok"] is False
    assert "stopped at session 2" in capsys.readouterr().err


def test_ctrl_c_at_the_prompt_stops_the_loop(env, monkeypatch, capsys):
    answers = iter(["", KeyboardInterrupt])

    def fake_input(prompt=""):
        answer = next(answers)
        if answer is KeyboardInterrupt:
            raise KeyboardInterrupt
        return answer

    monkeypatch.setattr("builtins.input", fake_input)
    assert collect.main([f"--plan={PLAN}"]) == 130
    assert [line["session"] for line in ledger(env)] == [1]
    assert "stopped at session 2" in capsys.readouterr().err


def test_next_and_number_still_run_one_session(env, monkeypatch):
    assert run(monkeypatch, "\n", "--session=next") == 0
    assert [line["session"] for line in ledger(env)] == [1]
    assert run(monkeypatch, "\n", "--session=3") == 0
    assert run(monkeypatch, "\n", "--session=next") == 0
    assert [line["session"] for line in ledger(env)] == [1, 3, 2]
    assert run(monkeypatch, "\n", "--session=3") == 1  # already complete


def test_dry_run_prints_remaining_sessions_and_touches_nothing(env, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert collect.main([f"--plan={PLAN}", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("$ ") == 6
    assert "Enter to record" not in out
    assert not env.exists()


def test_refuses_while_another_session_holds_the_arms(env, monkeypatch, capsys):
    from lab.core.hardware.lock import HardwareLock

    lock = HardwareLock(env / "locks" / "hardware.lock")
    lock.acquire("teleoperation")
    try:
        assert run(monkeypatch, "\n" * 6) == 1
    finally:
        lock.release()
    assert "teleoperation" in capsys.readouterr().err
    assert not (env / "raw" / "pick_place_v1.collection.jsonl").exists()
    assert run(monkeypatch, "\n") == 0  # the arms are free again: session 1, then EOF quits
    assert [line["session"] for line in ledger(env)] == [1]
