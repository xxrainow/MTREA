import json

import pytest

from robot.rollout_records import TrialRecorder


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
