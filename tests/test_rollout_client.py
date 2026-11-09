from unittest.mock import Mock

import pytest

from lab.core.hardware import HardwareBusy
from lab.core.hardware.lock import HardwareLock
from robot.rollout_client import create_locked_client

import logging

from robot.rollout_client import connect_policy, PolicyConnectionError


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
