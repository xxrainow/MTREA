import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robot.rollout_commands import CommandInbox, submit_command
from lab.core.hardware.lock import HardwareLock
from robot.rollout_worker import create_real_client, run_process
from test_rollout_client import FakeClientConfig, stoppable_client  # noqa: F401


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for rollout state")


def launch(tmp_path, client, lock, limit=30):
    root = tmp_path / "trial"
    results = []
    thread = Thread(target=lambda: results.append(run_process(
        client.config, trial_dir=root, time_limit_s=limit,
        client_factory=lambda *_: client, lock=lock,
    )), daemon=True)
    thread.start()
    wait_for(lambda: read(root / "status.json").get("state") in {"running", "stopped", "failed"})
    return root, thread, results


def result(root, command_id):
    return read(root / "command_results" / f"{command_id}.json")


def test_stop_keeps_owner_release_exits_and_rejects_early_release(tmp_path, stoppable_client):
    client, lock = stoppable_client
    root, thread, results = launch(tmp_path, client, lock)
    try:
        early = submit_command(root, "release")
        wait_for(lambda: result(root, early).get("state") == "rejected")
        stop = submit_command(root, "stop")
        wait_for(lambda: result(root, stop).get("state") == "completed")
        status = read(root / "status.json")
        assert status["state"] == "stopped"
        assert status["reason"] == "user_stop"
        assert status["hold_command_sent"] and status["workers_stopped"]
        assert status["process_alive"] and thread.is_alive() and lock.held
        client.robot.bus.disable_torque.assert_not_called()
    finally:
        submit_command(root, "stop")
        wait_for(lambda: read(root / "status.json").get("workers_stopped"))
        release = submit_command(root, "release")
        thread.join(5)
    assert not thread.is_alive() and results == [0]
    assert result(root, release)["state"] == "completed"
    assert not lock.held
    assert read(root / "status.json")["arm_released"]
    with pytest.raises(RuntimeError, match="not accepting"):
        submit_command(root, "stop")
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    assert events.count("release") == 1


def test_time_limit_and_release_failure_can_retry(tmp_path, stoppable_client):
    client, lock = stoppable_client
    root, thread, results = launch(tmp_path, client, lock, limit=0.02)
    wait_for(lambda: read(root / "status.json").get("workers_stopped"))
    assert read(root / "status.json")["reason"] == "time_limit"
    client.robot.bus.disable_torque.side_effect = OSError("USB unavailable")
    first = submit_command(root, "release")
    wait_for(lambda: result(root, first).get("state") == "failed")
    assert lock.held and thread.is_alive()
    assert read(root / "status.json")["release_error"] == "USB unavailable"
    client.robot.bus.disable_torque.side_effect = None
    second = submit_command(root, "release")
    thread.join(5)
    assert results == [0] and not lock.held
    assert result(root, second)["state"] == "completed"


def test_connection_failure_waits_for_release(tmp_path, stoppable_client):
    client, lock = stoppable_client
    client.start = lambda: False
    root, thread, results = launch(tmp_path, client, lock)
    wait_for(lambda: read(root / "status.json").get("workers_stopped"))
    assert thread.is_alive() and lock.held
    release = submit_command(root, "release")
    thread.join(5)
    assert results == [1] and not lock.held
    assert result(root, release)["state"] == "completed"


def test_stop_cancels_connecting_transport(tmp_path, stoppable_client):
    client, lock = stoppable_client
    entered = Event()
    cancelled = Event()

    def start():
        entered.set()
        assert cancelled.wait(5)
        return False

    client.start = start
    client.channel.close.side_effect = cancelled.set
    root = tmp_path / "trial"
    thread = Thread(target=lambda: run_process(
        client.config, trial_dir=root, time_limit_s=30,
        client_factory=lambda *_: client, lock=lock,
    ), daemon=True)
    thread.start()
    assert entered.wait(5)
    stop = submit_command(root, "stop")
    wait_for(lambda: result(root, stop).get("state") == "completed")
    assert read(root / "status.json")["reason"] == "user_stop"
    submit_command(root, "release")
    thread.join(5)
    assert not thread.is_alive() and not lock.held


@pytest.mark.parametrize("calibration", [{}, {"joint": "saved"}])
def test_real_factory_never_prompts_on_missing_or_mismatched_calibration(tmp_path, monkeypatch, calibration):
    robot = Mock()
    robot.calibration = calibration
    robot.connect.side_effect = lambda: robot.calibrate()
    lock = HardwareLock(tmp_path / "hardware.lock")

    class NativeClient:
        def __init__(self, config):
            self.robot = module.make_robot_from_config(config.robot)
            self.robot.connect()

    original = Mock(return_value=robot)
    module = SimpleNamespace(make_robot_from_config=original, RobotClient=NativeClient)
    monkeypatch.setitem(sys.modules, "lerobot.async_inference",
                        SimpleNamespace(robot_client=module))
    previous = Path.cwd()
    with pytest.raises(RuntimeError, match="calibration"):
        create_real_client(FakeClientConfig(), lock, tmp_path)
    assert module.make_robot_from_config is original
    assert Path.cwd() == previous and not lock.held
    if not calibration:
        robot.connect.assert_not_called()
    else:
        robot.disconnect.assert_called_once()


def test_initialization_failure_records_without_waiting(tmp_path):
    def fail(*_):
        raise RuntimeError("No calibration")
    assert run_process(FakeClientConfig(), trial_dir=tmp_path,
                       time_limit_s=1, client_factory=fail) == 1
    status = read(tmp_path / "status.json")
    assert status["state"] == "failed" and not status["process_alive"]
    assert "No calibration" in status["error"]
    assert (tmp_path / "session.json").exists()


def test_main_uses_labs_shared_data_root_for_hardware_lock(tmp_path, monkeypatch):
    from robot import rollout_worker
    monkeypatch.setattr(rollout_worker, "build_client_config", lambda _: FakeClientConfig())
    run = Mock(return_value=0)
    monkeypatch.setattr(rollout_worker, "run_process", run)
    assert rollout_worker.main([
        "--checkpoint=test", "--task=hold", f"--trial-dir={tmp_path / 'trial'}",
        f"--data-dir={tmp_path / 'lab-data'}", "--max-relative-target=5",
    ]) == 0
    assert run.call_args.kwargs["lock"].path == tmp_path / "lab-data" / "locks" / "hardware.lock"


def test_stale_malformed_and_duplicate_commands(tmp_path):
    inbox = CommandInbox(tmp_path, "current")
    (inbox.commands / "old.json").write_text(json.dumps({
        "id": "old", "action": "release", "trial_token": "previous",
    }))
    (inbox.commands / "bad.json").write_text("[]")
    (inbox.commands / "valid.json").write_text(json.dumps({
        "id": "valid", "action": "stop", "trial_token": "current",
    }))
    assert [command["id"] for command in inbox.poll()] == ["valid"]
    inbox.ack("valid", "completed")
    assert list(inbox.poll()) == []
    assert read(inbox.results / "old.json")["state"] == "rejected"
    assert read(inbox.results / "bad.json")["state"] == "rejected"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal integration")
def test_subprocess_signal_stop_requires_explicit_release(tmp_path):
    # Reuse the fake hardware fixture in a child: no serial ports or cameras opened.
    code = '''
import signal, sys
from pathlib import Path
from threading import Event
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_rollout_client import stoppable_client
from robot.rollout_worker import run_process
root = Path(sys.argv[1])
fixture = stoppable_client.__wrapped__(root)
client, lock = next(fixture)
stop = Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
try:
    code = run_process(client.config, trial_dir=root / "trial", time_limit_s=30,
                       client_factory=lambda *_: client, lock=lock, stop_requested=stop)
finally:
    fixture.close()
raise SystemExit(code)
'''
    process = subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    root = tmp_path / "trial"
    try:
        wait_for(lambda: read(root / "status.json").get("state") == "running")
        process.send_signal(signal.SIGTERM)
        wait_for(lambda: read(root / "status.json").get("workers_stopped"))
        assert process.poll() is None
        assert (tmp_path / "hardware.lock").exists()
        stop = submit_command(root, "stop")
        wait_for(lambda: result(root, stop).get("state") == "completed")
        release = submit_command(root, "release")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        assert result(root, release)["state"] == "completed"
        assert not (tmp_path / "hardware.lock").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
