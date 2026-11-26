from types import SimpleNamespace

import pytest

from robot.rollout_timing import RolloutTiming


def test_bounded_window_and_action_intervals():
    now = [0.0]
    timing = RolloutTiming(clock=lambda: now[0], window=2)
    for duration in (0.01, 0.02, 0.09):
        with timing.measure("camera"):
            now[0] += duration
    camera = timing.snapshot()["camera"]
    assert camera == {"count": 3, "recent_count": 2,
                      "last_ms": 90, "mean_ms": 55, "max_ms": 90}
    timing.action_completed()
    assert "action_interval" not in timing.snapshot()
    now[0] += 0.08
    timing.action_completed()
    assert timing.snapshot()["action_interval"]["last_ms"] == 80


def test_wrapped_io_preserves_arguments_results_and_failures():
    now = [0.0]
    timing = RolloutTiming(clock=lambda: now[0])

    def read(register, *, fail=False):
        now[0] += 0.05
        if fail:
            raise OSError("device failed")
        return {register: 123}

    device = SimpleNamespace(read=read)
    timing.wrap(device, "read", "serial_read")
    assert device.read("position") == {"position": 123}
    with pytest.raises(OSError, match="device failed"):
        device.read("position", fail=True)
    assert timing.snapshot()["serial_read"]["count"] == 2
    assert timing.snapshot()["serial_read"]["mean_ms"] == 50
