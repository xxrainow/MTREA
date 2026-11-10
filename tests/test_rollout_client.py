import json
import logging
import time
from dataclasses import dataclass, field
from queue import Queue
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lab.core.hardware import HardwareBusy
from lab.core.hardware.lock import HardwareLock
from robot.rollout_client import (
    PolicyConnectionError,
    connect_policy,
    create_locked_client,
    run_policy,
    run_recorded_policy,
)
from robot.rollout_records import TrialRecorder
from robot.rollout_preview import TrialPreview
from robot.rollout_stalls import QueueStallTracker


@dataclass
class FakeRobotConfig:
    id: str = "test-follower"
    type: str = "mock_robot"


@dataclass
class FakeClientConfig:
    robot: FakeRobotConfig = field(default_factory=FakeRobotConfig)
    pretrained_name_or_path: str = "fake-checkpoint"
    server_address: str = "127.0.0.1:8765"
    task: str = "Hold position"


@pytest.mark.parametrize("fail", [False, True])
def test_client_creation_holds_lock_and_cleans_up_failure(tmp_path, fail):
    lock = HardwareLock(tmp_path / "hardware.lock")
    robot = Mock()
    channel = Mock()

    class FakeClient:
        def __init__(self, config):
            assert lock.held
            self.robot = robot
            self.channel = channel
            if fail:
                raise RuntimeError("connection failed")

    try:
        if fail:
            with pytest.raises(RuntimeError, match="connection failed"):
                create_locked_client(
                    None, lock=lock, client_type=FakeClient
                )

            robot.disconnect.assert_called_once()
            channel.close.assert_called_once()
            assert not lock.held
            assert lock.holder() is None
        else:
            client = create_locked_client(
                None, lock=lock, client_type=FakeClient
            )

            assert client.robot is robot
            assert lock.held
            robot.disconnect.assert_not_called()
    finally:
        lock.release()


def test_busy_hardware_prevents_client_creation(tmp_path):
    path = tmp_path / "hardware.lock"
    holder = HardwareLock(path)
    contender = HardwareLock(path)

    class UnexpectedClient:
        def __init__(self, config):
            pytest.fail("Client must not be created while hardware is busy")

    holder.acquire("teleoperation")
    try:
        with pytest.raises(HardwareBusy):
            create_locked_client(
                None, lock=contender, client_type=UnexpectedClient
            )
    finally:
        holder.release()


@pytest.mark.parametrize("accepted", [True, False])
def test_policy_connection_reports_result_and_removes_handler(accepted):
    logger = logging.getLogger("tests.rollout.connection")
    original_handlers = tuple(logger.handlers)

    class FakeClient:
        def __init__(self):
            self.logger = logger

        def start(self):
            if not accepted:
                self.logger.error("refused: checkpoint mismatch")
            return accepted

    client = FakeClient()

    if accepted:
        connect_policy(client)
    else:
        with pytest.raises(
            PolicyConnectionError,
            match="refused: checkpoint mismatch",
        ):
            connect_policy(client)

    assert tuple(logger.handlers) == original_handlers


@pytest.fixture
def stoppable_client(tmp_path):
    lock = HardwareLock(tmp_path / "hardware.lock")
    robot = Mock()
    robot.config.num_read_retries = 2
    robot.bus.sync_read.return_value = {"gripper": 20.0}
    robot.bus.is_connected = True
    robot.bus.disconnect.side_effect = (
        lambda **kwargs: setattr(robot.bus, "is_connected", False)
    )

    camera = Mock()
    camera.is_connected = True
    camera.disconnect.side_effect = (
        lambda: setattr(camera, "is_connected", False)
    )
    robot.cameras = {"scene": camera}

    class FakeClient:
        def __init__(self, config):
            self.robot = robot
            self.channel = Mock()
            self.shutdown_event = Event()
            self.start_barrier = Barrier(2)
            self.logger = logging.getLogger("tests.rollout.running")
            self.config = FakeClientConfig()
            self.action_queue = Queue()
            self.action_queue_lock = Lock()

        @property
        def running(self):
            return not self.shutdown_event.is_set()

        def start(self):
            return True

        def receive_actions(self):
            self.start_barrier.wait()
            self.shutdown_event.wait()

        def control_loop(self, task):
            self.start_barrier.wait()
            while self.running:
                self.control_loop_action()
                self.control_loop_observation(task)
                self.shutdown_event.wait(0.01)

        def control_loop_action(self, verbose=False):
            self.robot.send_action({"gripper.pos": 30.0})

        def control_loop_observation(self, task, verbose=False):
            raw = self.robot.get_observation()
            captured = time.time()
            self.send_observation(SimpleNamespace(
                get_observation=lambda: raw,
                get_timestamp=lambda: captured,
            ))
            return raw

        def send_observation(self, observation):
            return True

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
            with self.action_queue_lock:
                for action in incoming_actions:
                    self.action_queue.put(action)

    client = create_locked_client(
        None, lock=lock, client_type=FakeClient
    )
    try:
        yield client, lock
    finally:
        lock.release()


def test_stop_holds_position_and_blocks_new_actions(stoppable_client):
    client, lock = stoppable_client

    client.control_loop_action()
    client.stop()
    client.stop()

    client.control_loop_action()
    client.control_loop_observation("Hold position")

    assert not client.running
    assert lock.held
    client.robot.send_action.assert_called_once()
    client.robot.get_observation.assert_not_called()
    client.robot.bus.sync_write.assert_called_once_with(
        "Goal_Position", {"gripper": 20.0}
    )
    client.robot.bus.disable_torque.assert_not_called()
    client.robot.disconnect.assert_not_called()


def test_release_requires_stop_and_releases_hardware(stoppable_client):
    client, lock = stoppable_client

    with pytest.raises(RuntimeError, match="Stop the client"):
        client.release_arm()

    client.stop()
    client.release_arm()
    client.release_arm()

    assert not lock.held
    assert lock.holder() is None
    client.robot.bus.disable_torque.assert_called_once_with(num_retry=5)
    client.robot.bus.disconnect.assert_called_once_with(
        disable_torque=False
    )
    client.robot.cameras["scene"].disconnect.assert_called_once()


def test_failed_hold_still_blocks_actions(stoppable_client):
    client, lock = stoppable_client
    client.robot.bus.sync_read.side_effect = OSError("read failed")

    with pytest.raises(OSError, match="read failed"):
        client.stop()

    client.control_loop_action()

    assert not client.running
    assert lock.held
    client.robot.send_action.assert_not_called()
    client.robot.bus.sync_write.assert_not_called()


@pytest.mark.parametrize("failure", ["torque", "camera"])
def test_failed_release_keeps_lock_and_can_retry(stoppable_client, failure):
    client, lock = stoppable_client
    client.stop()

    operation = (
        client.robot.bus.disable_torque
        if failure == "torque"
        else client.robot.cameras["scene"].disconnect
    )
    original_effect = operation.side_effect
    operation.side_effect = OSError("release failed")

    with pytest.raises(OSError, match="release failed"):
        client.release_arm()

    assert lock.held

    operation.side_effect = original_effect
    client.release_arm()

    assert not lock.held
    client.robot.bus.disconnect.assert_called_once_with(
        disable_torque=False
    )


def test_run_policy_stops_at_time_limit(stoppable_client):
    client, lock = stoppable_client

    reason = run_policy(
        client,
        task="Hold position",
        time_limit_s=0.2,
    )

    assert reason == "time_limit"
    assert lock.held
    assert not client.running
    assert all(not thread.is_alive() for thread in client._workers)
    client.robot.send_action.assert_called()
    client.robot.get_observation.assert_called()

    client.release_arm()
    assert not lock.held


def test_run_policy_accepts_stop_request(stoppable_client):
    client, lock = stoppable_client
    requested = Event()
    client.robot.send_action.side_effect = lambda action: requested.set()

    reason = run_policy(
        client,
        task="Hold position",
        time_limit_s=2.0,
        stop_requested=requested,
    )

    assert reason == "user_stop"
    assert lock.held
    assert not client.running
    assert all(not thread.is_alive() for thread in client._workers)


@pytest.mark.parametrize("failure", ["exception", "connection"])
def test_run_policy_stops_after_worker_failure(
    stoppable_client, monkeypatch, failure
):
    client, lock = stoppable_client

    def failing_receiver():
        client.start_barrier.wait()
        if failure == "exception":
            raise OSError("receiver failed")

        client.logger.error("Error receiving actions: connection lost")
        client.shutdown_event.wait()

    monkeypatch.setattr(client, "receive_actions", failing_receiver)
    expected = (
        OSError if failure == "exception" else PolicyConnectionError
    )

    with pytest.raises(expected):
        run_policy(
            client,
            task="Hold position",
            time_limit_s=2.0,
        )

    assert lock.held
    assert not client.running
    assert all(not thread.is_alive() for thread in client._workers)


def test_recorded_run_saves_terminal_state(stoppable_client, tmp_path):
    client, lock = stoppable_client
    recorder = TrialRecorder(tmp_path / "trial")

    run_recorded_policy(
        client,
        task="Hold position",
        time_limit_s=0.05,
        recorder=recorder,
    )

    status = json.loads(recorder.status_path.read_text())
    events = [
        json.loads(line)["event"]
        for line in recorder.events_path.read_text().splitlines()
    ]

    assert status["state"] == "stopped"
    assert status["reason"] == "time_limit"
    assert status["hold_command_sent"] is True
    assert status["workers_stopped"] is True
    assert status["arm_released"] is False
    assert lock.held
    assert events == ["created", "connect", "accept", "stop"]


@pytest.mark.parametrize("failure", ["connect", "connection", "hold"])
def test_recorded_run_saves_failure(
    stoppable_client, tmp_path, monkeypatch, failure
):
    client, lock = stoppable_client
    recorder = TrialRecorder(tmp_path / "trial")

    if failure == "connect":
        def reject():
            client.logger.error("refused: checkpoint mismatch")
            return False

        monkeypatch.setattr(client, "start", reject)

    elif failure == "connection":
        def lose_connection():
            client.start_barrier.wait()
            client.logger.error(
                "Error receiving actions: connection lost"
            )
            client.shutdown_event.wait()

        monkeypatch.setattr(
            client, "receive_actions", lose_connection
        )

    else:
        client.robot.bus.sync_read.side_effect = OSError("hold failed")

    expected_error = (
        OSError if failure == "hold" else PolicyConnectionError
    )

    with pytest.raises(expected_error):
        run_recorded_policy(
            client,
            task="Hold position",
            time_limit_s=0.05,
            recorder=recorder,
        )

    status = json.loads(recorder.status_path.read_text())
    assert status["state"] == (
        "aborted" if failure == "connection" else "failed"
    )
    assert status["error"]
    assert status["workers_stopped"] is True
    assert status["hold_command_sent"] is (failure != "hold")
    assert lock.held


def test_invalid_time_limit_still_stops_client(stoppable_client, tmp_path):
    client, lock = stoppable_client
    recorder = TrialRecorder(tmp_path / "trial")

    with pytest.raises(ValueError):
        run_recorded_policy(
            client,
            task="Hold position",
            time_limit_s=-1,
            recorder=recorder,
        )

    status = json.loads(recorder.status_path.read_text())
    assert status["state"] == "failed"
    assert not client.running
    assert lock.held


def test_client_counts_chunks_actions_and_latency(stoppable_client, monkeypatch):
    client, _ = stoppable_client
    monkeypatch.setattr("robot.rollout_client.time.time", lambda: 100.25)

    first = Mock()
    first.get_timestamp.return_value = 100.0
    client._aggregate_action_queues([first, first])

    metrics = client.metrics_snapshot()
    assert metrics["chunks_received"] == 1
    assert metrics["queue_size"] == 2
    assert metrics["last_latency_ms"] == pytest.approx(250.0)

    second = Mock()
    second.get_timestamp.return_value = 100.2
    client._aggregate_action_queues([second])
    client.control_loop_action()

    metrics = client.metrics_snapshot()
    assert metrics["chunks_received"] == 2
    assert metrics["actions_executed"] == 1
    assert metrics["last_latency_ms"] == pytest.approx(50.0)
    assert metrics["max_latency_ms"] == pytest.approx(250.0)


def test_failed_action_is_not_counted(stoppable_client):
    client, _ = stoppable_client
    client.robot.send_action.side_effect = OSError("send failed")

    with pytest.raises(OSError, match="send failed"):
        client.control_loop_action()

    assert client.metrics_snapshot()["actions_executed"] == 0


def test_running_metrics_are_written_before_stop(
    stoppable_client, tmp_path, monkeypatch
):
    client, _ = stoppable_client
    recorder = TrialRecorder(tmp_path / "trial")
    original_write = recorder.write_status
    observed = []

    def observe(*, state, **values):
        observed.append((state, values.get("actions_executed", 0)))
        original_write(state=state, **values)

    monkeypatch.setattr(recorder, "write_status", observe)

    run_recorded_policy(
        client,
        task="Hold position",
        time_limit_s=0.4,
        recorder=recorder,
    )

    assert any(
        state == "running" and actions > 0
        for state, actions in observed
    )

    status = json.loads(recorder.status_path.read_text())
    assert status["state"] == "stopped"
    assert status["actions_executed"] > 0


@pytest.mark.parametrize("resume", [False, True])
def test_recorded_stall_keeps_trial_alive_and_records_end(
    stoppable_client, tmp_path, monkeypatch, resume
):
    client, lock = stoppable_client
    client._stalls = QueueStallTracker(10)
    requested = Event()
    recorder = TrialRecorder(tmp_path / "trial")

    def waiting_control_loop(task):
        client.start_barrier.wait()
        if resume:
            client.shutdown_event.wait(0.04)
            client.control_loop_action()
            requested.set()
        client.shutdown_event.wait()

    monkeypatch.setattr(client, "control_loop", waiting_control_loop)
    reason = run_recorded_policy(
        client,
        task="Hold position",
        time_limit_s=0.08,
        recorder=recorder,
        stop_requested=requested,
    )

    assert reason == ("user_stop" if resume else "time_limit")
    status = json.loads(recorder.status_path.read_text())
    assert status["state"] == "stopped"
    assert status["stalls"] == 1
    assert status["stalling"] is False
    assert status["last_stall_ms"] >= 10
    assert lock.held
    assert not client.running

    phases = [
        event["details"]["phase"]
        for line in recorder.events_path.read_text().splitlines()
        if (event := json.loads(line))["event"] == "stall"
    ]
    assert phases == ["started", "resumed" if resume else "ended"]
    assert client.robot.send_action.call_count == int(resume)
    client.robot.bus.sync_write.assert_called_once_with(
        "Goal_Position", {"gripper": 20.0}
    )


def test_recorded_run_publishes_existing_observations_and_session(
    stoppable_client, tmp_path
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from robot.config import CAMERAS

    client, lock = stoppable_client
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    client.robot.get_observation.return_value = {
        camera.name: image for camera in CAMERAS
    }
    recorder = TrialRecorder(tmp_path / "trial")
    preview = TrialPreview(recorder.root)
    run_recorded_policy(
        client, task="Effective instruction", time_limit_s=0.25,
        recorder=recorder, preview=preview,
    )
    status = json.loads(recorder.status_path.read_text())
    session = json.loads(recorder.session_path.read_text())
    assert status["state"] == "stopped"
    assert status["preview_active"] is False
    assert status["preview_error"] is None
    assert session["task"] == "Effective instruction"
    assert session["configuration"]["robot"]["type"] == "mock_robot"
    for camera in CAMERAS:
        assert (preview.root / f"{camera.name}.jpg").is_file()
    assert client._frame_publisher is None
    assert lock.held


def test_preview_encoding_error_does_not_abort_policy(stoppable_client, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from robot.config import CAMERAS

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        COLOR_RGB2BGR=1, IMWRITE_JPEG_QUALITY=2,
        cvtColor=lambda frame, _: frame, imencode=lambda *_: (False, None),
    ))
    frame = SimpleNamespace(shape=(2, 2, 3))
    frame.copy = lambda: frame
    client, lock = stoppable_client
    client.robot.get_observation.return_value = {
        camera.name: frame for camera in CAMERAS
    }
    recorder = TrialRecorder(tmp_path / "trial")
    preview = TrialPreview(recorder.root)
    reason = run_recorded_policy(
        client, task="Hold position", time_limit_s=0.1,
        recorder=recorder, preview=preview,
    )
    status = json.loads(recorder.status_path.read_text())
    assert reason == "time_limit"
    assert status["state"] == "stopped"
    assert "Cannot encode" in status["preview_error"]
    assert status["actions_executed"] > 0
    assert lock.held


def test_preview_receives_capture_before_transport(stoppable_client, monkeypatch):
    client, _ = stoppable_client
    publisher = Mock()
    client._frame_publisher = publisher
    raw = {"observation": "already captured"}
    observation = SimpleNamespace(
        get_observation=lambda: raw,
        get_timestamp=lambda: 123.0,
    )

    def transport(self, received):
        publisher.submit.assert_called_once_with(raw, captured_at=123.0)
        assert received is observation
        return True

    monkeypatch.setattr(type(client).__bases__[0], "send_observation", transport)
    assert client.send_observation(observation) is True


def test_session_write_failure_stops_before_server_connection(
    stoppable_client, tmp_path, monkeypatch
):
    client, lock = stoppable_client
    recorder = TrialRecorder(tmp_path / "trial")
    start = Mock()
    monkeypatch.setattr(client, "start", start)

    def fail_write(metadata):
        raise OSError("session write failed")

    monkeypatch.setattr(recorder, "write_session", fail_write)
    with pytest.raises(OSError, match="session write failed"):
        run_recorded_policy(
            client, task="Hold position", time_limit_s=1,
            recorder=recorder,
        )
    start.assert_not_called()
    assert not client.running
    assert lock.held
    assert json.loads(recorder.status_path.read_text())["state"] == "failed"
