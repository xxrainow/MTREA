import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from robot.rollout_records import TrialRecorder, build_trial_metadata


@dataclass
class RobotConfig:
    id: str = "synthetic-arm"
    calibration_dir: Path = Path("synthetic-calibration")


@dataclass
class ClientConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    pretrained_name_or_path: str = "synthetic-checkpoint"
    server_address: str = "127.0.0.1:8765"
    task: str = "Original instruction"


def test_trial_keeps_event_history_and_latest_status(tmp_path):
    recorder = TrialRecorder(tmp_path / "trial")
    original_events = recorder.events_path.read_bytes()

    recorder.write_status(state="running", chunks_received=3)
    recorder.event("running")
    recorder.event("stop", reason="time_limit")
    recorder.write_status(state="stopped", reason="time_limit")

    status = json.loads(recorder.status_path.read_text())
    assert status["state"] == "stopped"
    assert status["reason"] == "time_limit"
    assert status["elapsed_s"] >= 0

    assert recorder.events_path.read_bytes().startswith(original_events)

    events = [
        json.loads(line)
        for line in recorder.events_path.read_text().splitlines()
    ]
    assert [item["event"] for item in events] == [
        "created", "running", "stop"
    ]
    assert events[-1]["details"]["reason"] == "time_limit"


def test_existing_trial_is_refused_without_changes(tmp_path):
    trial_dir = tmp_path / "trial"
    TrialRecorder(trial_dir)
    before = {
        path.name: path.read_bytes()
        for path in trial_dir.iterdir()
    }

    with pytest.raises(FileExistsError):
        TrialRecorder(trial_dir)

    after = {
        path.name: path.read_bytes()
        for path in trial_dir.iterdir()
    }
    assert after == before


def test_invalid_number_preserves_previous_status(tmp_path):
    recorder = TrialRecorder(tmp_path / "trial")
    before = recorder.status_path.read_bytes()

    with pytest.raises(ValueError):
        recorder.write_status(
            state="running",
            latency_ms=float("nan"),
        )

    assert recorder.status_path.read_bytes() == before


def test_session_records_resolved_configuration_provenance_and_effective_task(
    tmp_path, monkeypatch
):
    calls = []

    def git(argv, **kwargs):
        calls.append((argv, kwargs["cwd"]))
        text = "a" * 40 if argv[1] == "rev-parse" else "?? new_file.py\n"
        return subprocess.CompletedProcess(argv, 0, stdout=text)

    monkeypatch.setattr("robot.rollout_records.subprocess.run", git)
    monkeypatch.setattr("robot.rollout_records.version", lambda _: "0.6.1")
    recorder = TrialRecorder(tmp_path / "trial")
    config = ClientConfig()
    recorder.write_session(build_trial_metadata(
        config, time_limit_s=12.5, repo_root=tmp_path, task="Effective instruction",
    ))
    saved = json.loads(recorder.session_path.read_text())
    assert saved["git_commit"] == "a" * 40
    assert saved["git_dirty"] is True
    assert saved["lerobot_version"] == "0.6.1"
    assert saved["checkpoint"] == config.pretrained_name_or_path
    assert saved["task"] == "Effective instruction"
    assert saved["time_limit_s"] == 12.5
    assert saved["configuration"]["robot"]["calibration_dir"] == "synthetic-calibration"
    assert config.task == "Original instruction"
    assert len(calls) == 2
    assert all(root == tmp_path for _, root in calls)


def test_unavailable_git_is_explicit_and_session_cannot_be_rewritten(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("robot.rollout_records.subprocess.run", unavailable)
    recorder = TrialRecorder(tmp_path)
    recorder.write_session(build_trial_metadata(ClientConfig(), time_limit_s=5))
    before = recorder.session_path.read_bytes()
    saved = json.loads(before)
    assert saved["git_commit"] is None
    assert saved["git_dirty"] is None
    with pytest.raises(FileExistsError):
        recorder.write_session({"changed": True})
    assert recorder.session_path.read_bytes() == before


def test_session_without_other_trial_files_is_also_preserved(tmp_path):
    session = tmp_path / "session.json"
    session.write_text('{"prior":true}')
    with pytest.raises(FileExistsError):
        TrialRecorder(tmp_path)
    assert session.read_text() == '{"prior":true}'
    assert not (tmp_path / "events.jsonl").exists()
