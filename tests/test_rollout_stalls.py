import pytest

from robot.rollout_stalls import QueueStallTracker


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_no_stalls_before_start_or_below_threshold():
    clock = Clock()
    tracker = QueueStallTracker(500, clock)
    clock.now = 10.0
    assert tracker.poll() == []
    tracker.start()
    clock.now = 10.49
    assert tracker.poll() == []
    assert tracker.snapshot()["stalls"] == 0


def test_initial_wait_is_counted_once_and_duration_updates():
    clock = Clock()
    tracker = QueueStallTracker(500, clock)
    tracker.start()
    clock.now = 0.5
    assert tracker.poll() == [{"phase": "started", "duration_ms": 500.0}]
    clock.now = 1.25
    assert tracker.poll() == []
    snapshot = tracker.snapshot()
    assert snapshot["stalls"] == 1
    assert snapshot["stalling"] is True
    assert snapshot["stall_ms"] == pytest.approx(1250.0)
    assert snapshot["max_stall_ms"] == pytest.approx(1250.0)


def test_actions_resume_stall_and_a_later_gap_is_separate():
    clock = Clock()
    tracker = QueueStallTracker(500, clock)
    tracker.start()
    clock.now = 0.6
    tracker.poll()
    clock.now = 0.8
    tracker.action_executed()
    assert tracker.poll() == [{"phase": "resumed", "duration_ms": 800.0}]
    assert tracker.snapshot()["stalling"] is False
    clock.now = 1.4
    assert tracker.poll()[0]["phase"] == "started"
    assert tracker.snapshot()["stalls"] == 2
    assert tracker.snapshot()["max_stall_ms"] == pytest.approx(800.0)


def test_recovery_between_polls_preserves_both_events():
    clock = Clock()
    tracker = QueueStallTracker(500, clock)
    tracker.start()
    clock.now = 0.75
    tracker.action_executed()
    events = tracker.poll()
    assert [event["phase"] for event in events] == ["started", "resumed"]
    assert tracker.snapshot()["last_stall_ms"] == pytest.approx(750.0)


def test_stop_closes_unresolved_wait_once_without_reporting_recovery():
    clock = Clock()
    tracker = QueueStallTracker(500, clock)
    tracker.start()
    clock.now = 0.75
    tracker.finish()
    tracker.finish()
    clock.now = 10.0
    tracker.action_executed()
    events = tracker.poll()
    assert [event["phase"] for event in events] == ["started", "ended"]
    snapshot = tracker.snapshot()
    assert snapshot["stalls"] == 1
    assert snapshot["stalling"] is False
    assert snapshot["stall_ms"] == 0
    assert snapshot["last_stall_ms"] == pytest.approx(750.0)


def test_regular_actions_do_not_accumulate_into_a_stall():
    clock = Clock()
    tracker = QueueStallTracker(500, clock)
    tracker.start()
    for tick in range(1, 30):
        clock.now = tick * 0.1
        tracker.action_executed()
        assert tracker.poll() == []
    assert tracker.snapshot()["stalls"] == 0


@pytest.mark.parametrize("threshold", [None, True, 0, -1, float("nan"), float("inf"), "500"])
def test_invalid_threshold_is_refused(threshold):
    with pytest.raises(ValueError):
        QueueStallTracker(threshold)
