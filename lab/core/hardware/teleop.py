"""Leader -> follower teleoperation in a worker thread, at robot.config.FPS.

start() opens both arms before returning, so a wrong port or a missing
calibration reaches the caller instead of dying in the thread (as in LeLab).
The worker then owns both ports until stop(): each tick reads the leader,
sends it to the follower (LeRobot clamps each step to max_relative_target from
the follower yaml), reads the follower back and publishes
{type: "teleop", t, leader: {joint: pos}, follower: {joint: pos}}.

On stop the follower's torque is switched off: lerobot v0.6.1
SOFollowerConfig.disable_torque_on_disconnect defaults to True and
disconnect() honours it. The follower goes limp and drops unless it rests on
something, so bring it to its rest pose with the leader before stopping. The
leader's torque is off throughout.
"""

from __future__ import annotations

import collections
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from lab.core.hardware import JointStream, SessionStateError
from lab.core.hardware.devices import CONFIGS_DIR, load_arm_config
from lab.core.hardware.lock import HardwareLock
from robot.config import FPS

TORQUE_ON_STOP = ("follower torque turns off when teleoperation stops (lerobot "
                  "disable_torque_on_disconnect=True): the arm goes limp; rest it first")


class TeleopSession:
    def __init__(self, backend: Any, lock: HardwareLock, stream: JointStream,
                 configs_dir: Path = CONFIGS_DIR, fps: int = FPS):
        self.backend, self.lock, self.stream = backend, lock, stream
        self.configs_dir, self.fps = configs_dir, fps
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None
        self._ticks: collections.deque[float] = collections.deque()
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._max_relative_target = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> dict:
        with self._mutex:
            if self._running:
                raise SessionStateError("teleoperation is already running")
            follower_cfg = load_arm_config("follower", self.configs_dir)
            leader_cfg = load_arm_config("leader", self.configs_dir)
            self.lock.acquire("teleoperation")
            follower = leader = None
            try:
                follower = self.backend.teleop_arm(follower_cfg)
                follower.connect()
                leader = self.backend.teleop_arm(leader_cfg)
                leader.connect()
            except BaseException:
                for device in (leader, follower):
                    if device is not None:
                        device.disconnect()
                self.lock.release()
                raise
            self._stop.clear()
            self._ticks.clear()
            self._error, self._stopped_at = None, None
            self._started_at = time.time()
            self._max_relative_target = follower_cfg.max_relative_target
            self._running = True
            self._thread = threading.Thread(target=self._run, args=(follower, leader),
                                            name="teleop", daemon=True)
            self._thread.start()
        return self.status()

    def stop(self, timeout_s: float = 5.0) -> dict:
        with self._mutex:
            if not self._running:
                raise SessionStateError("teleoperation is not running")
            self._stop.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        return self.status()

    def shutdown(self) -> None:
        if self._running:
            self.stop()

    def status(self) -> dict:
        now = time.monotonic()
        with self._mutex:
            ticks = [t for t in self._ticks if now - t <= 1.0]
            return {
                "backend": self.backend.name,
                "running": self._running,
                "fps": len(ticks) if self._running else 0,
                "target_fps": self.fps,
                "error": self._error,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "max_relative_target": self._max_relative_target,
                "torque_on_stop": TORQUE_ON_STOP,
            }

    def _run(self, follower: Any, leader: Any) -> None:
        period = 1.0 / self.fps
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                leader_pos = leader.read()
                follower.send(leader_pos)
                follower_pos = follower.read()
                self.stream.publish({"type": "teleop", "t": time.time(),
                                     "leader": leader_pos, "follower": follower_pos})
                now = time.monotonic()
                with self._mutex:
                    self._ticks.append(now)
                    while self._ticks and now - self._ticks[0] > 1.0:
                        self._ticks.popleft()
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_tick = time.perf_counter()  # fell behind; don't try to catch up
        except Exception as exc:
            self._error = f"teleoperation stopped: {exc.__class__.__name__}: {exc}"
            traceback.print_exc()
        finally:
            try:
                leader.disconnect()
                follower.disconnect()
            finally:
                self.lock.release()
                with self._mutex:
                    self._running = False
                    self._stopped_at = time.time()
