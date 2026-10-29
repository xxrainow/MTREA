import json
from pathlib import Path
import sys
from time import monotonic, sleep

import pytest

from lab.core.recording_telemetry import TelemetryReader, TelemetryWriter
from lab.core.recording_worker import read_recording_command, recording_commands
from lab.core.recorder import RecordingProcess


def write_retry(root, phase_id):
    (root / "command.json").write_text(json.dumps({
        "id": "retry-command",
        "action": "retry_episode",
        "expected_phase": "recording",
        "expected_episode": 1,
        "expected_phase_id": phase_id,
    }))


def test_retry_file_sets_lerobot_flags_and_does_not_leak_exit_to_reset(tmp_path):
    writer = TelemetryWriter(tmp_path)
    writer.stage("recording", 30)
    write_retry(tmp_path, writer.phase_id)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    with recording_commands(
        tmp_path, phase="recording", episode=1,
        phase_id=writer.phase_id, events=events,
    ):
        deadline = monotonic() + 2
        while not events["exit_early"]:
            assert monotonic() < deadline
            sleep(0.001)
        assert events["rerecord_episode"]

    assert not events["exit_early"]
    assert events["rerecord_episode"]
    assert not (tmp_path / "command.json").exists()
    result = json.loads((tmp_path / "command_result.json").read_text())
    assert result["status"] == "accepted"


def test_retry_from_previous_attempt_is_rejected_with_same_phase_and_episode(tmp_path):
    writer = TelemetryWriter(tmp_path)
    writer.stage("recording", 30)
    old_phase_id = writer.phase_id
    writer.stage("resetting", 0)
    writer.stage("recording", 30)
    assert writer.phase_id != old_phase_id
    assert TelemetryReader(tmp_path).progress()["phase_id"] == writer.phase_id
    write_retry(tmp_path, old_phase_id)

    assert read_recording_command(
        tmp_path, phase="recording", episode=1, phase_id=writer.phase_id,
    ) is None
    result = json.loads((tmp_path / "command_result.json").read_text())
    assert result["status"] == "rejected"
    assert "attempt has changed" in result["error"]


def test_command_watcher_failure_interrupts_recording_and_propagates(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Command channel failed")

    monkeypatch.setattr("lab.core.recording_worker.read_recording_command", fail)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    with pytest.raises(OSError, match="Command channel failed"):
        with recording_commands(
            tmp_path, phase="recording", episode=1,
            phase_id="phase-1", events=events,
        ):
            deadline = monotonic() + 2
            while not events["exit_early"]:
                assert monotonic() < deadline
                sleep(0.001)
    assert not events["exit_early"]


def test_retry_reaches_separate_worker_through_command_file(tmp_path, monkeypatch):
    child = tmp_path / "command_worker.py"
    child.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "from lab.core.recording_telemetry import TelemetryWriter\n"
        "from lab.core.recording_worker import recording_commands\n"
        "root = Path(sys.argv[1])\n"
        "writer = TelemetryWriter(root)\n"
        "events = dict(exit_early=False, rerecord_episode=False, stop_recording=False)\n"
        "writer.stage('recording', 30)\n"
        "with recording_commands(root, phase='recording', episode=1, events=events, phase_id=writer.phase_id):\n"
        "    deadline = time.monotonic() + 3\n"
        "    while not events['exit_early']:\n"
        "        if time.monotonic() > deadline: raise TimeoutError('No retry command')\n"
        "        time.sleep(0.001)\n"
        "(root / 'events.json').write_text(json.dumps(events))\n"
        "writer.stage('completed')\n"
    )
    import lab.core.recorder as recorder

    original_popen = recorder.subprocess.Popen

    def popen(command, **options):
        return original_popen([sys.executable, str(child), command[3]], **options)

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setattr(recorder.subprocess, "Popen", popen)
    process = RecordingProcess(["lerobot-record"], telemetry=True)
    root = Path(process._telemetry_dir.name)
    try:
        deadline = monotonic() + 3
        while process.progress().get("episode_phase") != "recording":
            assert monotonic() < deadline
            sleep(0.005)
        phase_id = process.progress()["phase_id"]
        command_id = process.send_command(
            "retry_episode", expected_phase="recording",
            expected_episode=1, expected_phase_id=phase_id,
        )
        assert process.wait() == 0
        result = json.loads((root / "command_result.json").read_text())
        assert result["id"] == command_id
        assert result["status"] == "accepted"
        events = json.loads((root / "events.json").read_text())
        assert events["rerecord_episode"]
        assert not events["exit_early"]
    finally:
        process.request_stop()
        process.wait()
        process.close()
    assert not root.exists()
