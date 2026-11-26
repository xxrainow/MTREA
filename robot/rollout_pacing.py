"""Keep control ticks on monotonic deadlines without catch-up action bursts."""

import math
import time


class ControlPacer:
    def __init__(self, fps, *, clock=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self._clock = clock or time.monotonic
        self.period = 1.0 / fps
        self.deadline = self._clock()
        self.skipped = 0

    def lateness(self):
        return max(0.0, self._clock() - self.deadline)

    def wait_next(self, stop):
        self.deadline += self.period
        now = self._clock()
        if now > self.deadline:
            skipped = math.floor((now - self.deadline) / self.period) + 1
            self.deadline += skipped * self.period
            self.skipped += skipped
        return stop.wait(max(0.0, self.deadline - now))
