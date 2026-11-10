import math
import time
from collections.abc import Callable
from threading import Lock


class QueueStallTracker:
    """Measure action gaps without doing I/O in the control thread."""

    def __init__(self, threshold_ms: int, clock: Callable[[], float] = time.monotonic):
        if (
            isinstance(threshold_ms, bool)
            or not isinstance(threshold_ms, (int, float))
            or not math.isfinite(threshold_ms)
            or threshold_ms <= 0
        ):
            raise ValueError("threshold_ms must be a finite positive number")
        self._threshold_ms = threshold_ms
        self._clock = clock
        self._lock = Lock()
        self._running = False
        self._last_action = None
        self._stalling = False
        self._stalls = 0
        self._last_stall_ms = None
        self._max_stall_ms = 0.0
        self._events: list[dict] = []

    def start(self) -> None:
        with self._lock:
            if self._last_action is None:
                self._last_action = self._clock()
                self._running = True

    def _gap_ms(self, now: float) -> float:
        if self._last_action is None:
            return 0.0
        return max(0.0, (now - self._last_action) * 1000)

    def _check(self, now: float) -> None:
        if self._running and not self._stalling:
            duration = self._gap_ms(now)
            if duration >= self._threshold_ms:
                self._stalling = True
                self._stalls += 1
                self._events.append({"phase": "started", "duration_ms": duration})

    def _close(self, now: float, phase: str) -> None:
        if self._stalling:
            duration = self._gap_ms(now)
            self._last_stall_ms = duration
            self._max_stall_ms = max(self._max_stall_ms, duration)
            self._events.append({"phase": phase, "duration_ms": duration})
            self._stalling = False

    def action_executed(self) -> None:
        with self._lock:
            if not self._running:
                return
            now = self._clock()
            self._check(now)
            self._close(now, "resumed")
            self._last_action = now

    def poll(self) -> list[dict]:
        with self._lock:
            self._check(self._clock())
            events, self._events = self._events, []
            return events

    def finish(self) -> None:
        with self._lock:
            if self._running:
                now = self._clock()
                self._check(now)
                self._close(now, "ended")
                self._running = False

    def snapshot(self) -> dict:
        with self._lock:
            current = self._gap_ms(self._clock()) if self._stalling else 0.0
            return {
                "stalls": self._stalls,
                "stalling": self._stalling,
                "stall_ms": current,
                "last_stall_ms": self._last_stall_ms,
                "max_stall_ms": max(self._max_stall_ms, current),
            }
