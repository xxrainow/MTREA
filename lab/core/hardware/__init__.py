"""SO-101 hardware sessions: port detection, calibration, teleoperation.

Knows nothing about HTTP (lab/server/hardware_api.py wraps service.py) and
imports lerobot only inside the real backend (devices.py), so fake mode, the
tests and the server start without lerobot or arms. Only one session holds the
arms at a time (lock.py). MTREA_HARDWARE=fake selects fake.py; it is never the
default.
"""

from __future__ import annotations

import threading
from typing import Callable

ARMS = ("follower", "leader")


class HardwareError(RuntimeError):
    """A hardware operation failed. The message is meant for the operator."""


class HardwareBusy(HardwareError):
    """Another session or process holds the arms."""


class SessionStateError(HardwareError):
    """The request does not fit the session's current state."""


class PortDetectionError(HardwareError):
    """Unplug-to-detect found no port, or more than one."""


def check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise HardwareError(f"arm must be one of {ARMS}, got {arm!r}")
    return arm


class JointStream:
    """Fan-out of joint frames from a session's worker thread.

    publish() runs on the worker thread and calls every subscriber inline, so
    subscribers must only hand the frame off (e.g. call_soon_threadsafe) and
    never block. A failing subscriber is dropped, never the worker.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(self, frame: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(frame)
            except Exception:
                with self._lock:
                    if callback in self._subscribers:
                        self._subscribers.remove(callback)
