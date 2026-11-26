from dataclasses import dataclass
from threading import Event, Thread

import pytest

from robot.rollout_sender import LatestObservationSender


@dataclass
class Observation:
    observation: dict
    timestep: int
    timestamp: float = 123.0
    must_go: bool = False


def test_latest_pending_snapshot_replaces_old_and_preserves_must_go():
    received = []
    first_entered, unblock, second_sent = Event(), Event(), Event()

    def send(observation):
        received.append(observation)
        if len(received) == 1:
            first_entered.set()
            assert unblock.wait(3)
        else:
            second_sent.set()
        return True

    sender = LatestObservationSender(send, max_age_s=2)
    sender.submit(Observation({"pixels": [1]}, 1))
    thread = Thread(target=sender.run)
    thread.start()
    try:
        assert first_entered.wait(2)
        sender.submit(Observation({"pixels": [2]}, 2, must_go=True))
        latest = Observation({"pixels": [3]}, 3)
        sender.submit(latest)
        latest.observation["pixels"][0] = 999
        assert sender.snapshot()["pending"] == 1
        assert sender.snapshot()["replaced"] == 1
        unblock.set()
        assert second_sent.wait(2)
    finally:
        sender.close()
        unblock.set()
        thread.join(2)
    assert not thread.is_alive()
    assert [obs.timestep for obs in received] == [1, 3]
    assert received[1].observation == {"pixels": [3]}
    assert received[1].must_go and not latest.must_go
    assert received[1].timestamp == 123.0
    assert sender.submit(latest) is False


def test_stale_pending_observation_is_dropped_and_force_flag_survives():
    now = [0.0]
    received = []
    sent = Event()
    stale_checked = Event()
    calls = [0]

    def clock():
        calls[0] += 1
        if calls[0] == 2:
            stale_checked.set()
        return now[0]

    def send(observation):
        received.append(observation)
        sent.set()
        return True

    sender = LatestObservationSender(send, max_age_s=2, clock=clock)
    sender.submit(Observation({}, 1, must_go=True))
    now[0] = 3
    thread = Thread(target=sender.run)
    thread.start()
    try:
        assert stale_checked.wait(2)
        sender.submit(Observation({}, 2))
        assert sent.wait(2)
    finally:
        sender.close()
        thread.join(2)
    assert [obs.timestep for obs in received] == [2]
    assert received[0].must_go
    assert sender.snapshot()["stale"] == 1


def test_transport_exception_closes_sender_and_discards_pending():
    sender = None

    def fail(observation):
        sender.submit(Observation({}, 2))
        raise OSError("connection failed")

    sender = LatestObservationSender(fail, max_age_s=2)
    sender.submit(Observation({}, 1))
    with pytest.raises(OSError, match="connection failed"):
        sender.run()
    assert sender.snapshot()["closed"]
    assert sender.snapshot()["pending"] == 0


def test_refill_discards_pending_and_rejects_captures_from_previous_generation():
    received = []
    entered, unblock, latest_sent = Event(), Event(), Event()

    def send(observation):
        received.append(observation)
        if len(received) == 1:
            entered.set()
            assert unblock.wait(3)
        else:
            latest_sent.set()
        return True

    sender = LatestObservationSender(send, max_age_s=2)
    generation = sender.generation()
    sender.submit(Observation({}, 1), generation=generation)
    thread = Thread(target=sender.run)
    thread.start()
    try:
        assert entered.wait(2)
        sender.submit(Observation({}, 2, must_go=True), generation=generation)
        sender.invalidate_pending()
        assert sender.snapshot()["pending"] == 0
        assert not sender.submit(Observation({}, 3), generation=generation)
        assert sender.submit(Observation({}, 4), generation=sender.generation())
        unblock.set()
        assert latest_sent.wait(2)
    finally:
        sender.close()
        unblock.set()
        thread.join(2)
    assert [obs.timestep for obs in received] == [1, 4]
    assert not received[1].must_go
    assert sender.snapshot()["superseded"] == 2


def test_sender_reports_local_queue_wait_without_changing_timestamp():
    now = [0.0]
    waits, received = [], []
    sender = None

    def send(observation):
        received.append(observation)
        sender.close()
        return True

    sender = LatestObservationSender(send, max_age_s=2, clock=lambda: now[0],
                                     on_queue_wait=waits.append)
    sender.submit(Observation({}, 1, timestamp=42))
    now[0] = 0.12
    sender.run()
    assert waits == [0.12]
    assert received[0].timestamp == 42
