import pytest

from robot.rollout_pacing import ControlPacer


class FakeClock:
    def __init__(self, overshoot=0):
        self.now = 0.0
        self.overshoot = overshoot
        self.waits = []

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds + self.overshoot
        return False


def test_wait_overshoot_does_not_accumulate_into_control_rate():
    clock = FakeClock(overshoot=0.003)
    pacer = ControlPacer(30, clock=clock)
    ticks = []
    for _ in range(301):
        ticks.append(clock())
        clock.now += 0.002
        pacer.wait_next(clock)
    assert ticks[-1] == pytest.approx(10.003)
    assert (ticks[-1] - ticks[1]) / 299 == pytest.approx(1 / 30)
    assert pacer.skipped == 0


def test_long_work_skips_deadlines_without_catch_up_bursts():
    clock = FakeClock()
    pacer = ControlPacer(30, clock=clock)
    clock.now = 0.09
    pacer.wait_next(clock)
    assert pacer.skipped == 2
    assert clock.now == pytest.approx(0.1)
    assert clock.waits[-1] == pytest.approx(0.01)
    clock.now += 0.001
    pacer.wait_next(clock)
    assert clock.now == pytest.approx(4 / 30)


def test_stop_interrupts_deadline_wait():
    class Stopped:
        def wait(self, seconds):
            return True

    pacer = ControlPacer(30, clock=lambda: 0)
    assert pacer.wait_next(Stopped()) is True


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan")])
def test_invalid_fps_is_refused(fps):
    with pytest.raises(ValueError, match="fps"):
        ControlPacer(fps)
