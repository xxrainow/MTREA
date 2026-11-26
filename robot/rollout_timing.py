"""Bounded in-memory diagnostics; snapshots are written by the rollout monitor."""

import time
from collections import deque
from contextlib import contextmanager
from functools import wraps
from threading import Lock


class RolloutTiming:
    def __init__(self, *, clock=time.perf_counter, window=256):
        self._clock = clock
        self._window = window
        self._lock = Lock()
        self._samples = {}
        self._counts = {}
        self._last_action = None

    def record(self, name, seconds):
        with self._lock:
            self._samples.setdefault(name, deque(maxlen=self._window)).append(seconds * 1000)
            self._counts[name] = self._counts.get(name, 0) + 1

    @contextmanager
    def measure(self, name):
        started = self._clock()
        try:
            yield
        finally:
            self.record(name, self._clock() - started)

    def action_completed(self):
        now = self._clock()
        with self._lock:
            previous, self._last_action = self._last_action, now
        if previous is not None:
            self.record("action_interval", now - previous)

    def wrap(self, owner, method, name):
        original = getattr(owner, method, None)
        if not callable(original):
            return

        @wraps(original)
        def measured(*args, **kwargs):
            with self.measure(name):
                return original(*args, **kwargs)

        setattr(owner, method, measured)

    def snapshot(self):
        with self._lock:
            samples = {name: (self._counts[name], tuple(values))
                       for name, values in self._samples.items()}
        return {
            name: {"count": count, "recent_count": len(values),
                   "last_ms": round(values[-1], 3),
                   "mean_ms": round(sum(values) / len(values), 3),
                   "max_ms": round(max(values), 3)}
            for name, (count, values) in samples.items()
        }
