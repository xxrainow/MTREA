"""Fake arms for home development (MTREA_HARDWARE=fake). Never the default.

Same interface as devices.RealBackend. Ports: a fixed list in which the
detected arm's port disappears about a second after detection starts.
Calibration: smooth raw encoder sweeps around mid-range that advance on Next.
Teleop: a leader that waves every joint, a follower that follows it (clamped
by max_relative_target). Frames carry backend "fake" in every status, and a
fake calibration is written as <id>.fake.json, a name lerobot never reads, so
it cannot replace a real calibration.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Callable

from lab.core.hardware import check_arm
from lab.core.hardware.devices import ArmConfig, calibration_path
from robot.config import JOINT_ORDER

FAKE_PORTS = {"follower": "/dev/tty.fake-follower", "leader": "/dev/tty.fake-leader"}
_OTHER_PORTS = ["/dev/tty.Bluetooth-Incoming-Port", "/dev/tty.debug-console"]
UNPLUG_AFTER_S = 1.0

Trajectory = Callable[[float], dict[str, int]]


def default_trajectory(t: float) -> dict[str, int]:
    """Raw 12-bit positions sweeping +-900 steps around mid-range, each joint out of phase."""
    return {joint: int(2047 + 900 * math.sin(2 * math.pi * t / 2.0 + i)) for i, joint in enumerate(JOINT_ORDER)}


class FakeCalibrationArm:
    def __init__(self, cfg: ArmConfig, trajectory: Trajectory):
        self.cfg, self.trajectory = cfg, trajectory
        self._t0 = 0.0

    def connect(self) -> None:
        time.sleep(0.05)
        self._t0 = time.monotonic()

    def read_raw(self) -> dict[str, int]:
        return self.trajectory(time.monotonic() - self._t0)

    def set_half_turn_homings(self) -> dict[str, int]:
        return {joint: pos - 2047 for joint, pos in self.read_raw().items()}

    def save(self, homing: dict[str, int], mins: dict[str, int], maxes: dict[str, int]) -> Path:
        path = fake_calibration_path(self.cfg.arm, self.cfg.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The shape lerobot's _save_calibration() writes (draccus MotorCalibration).
        calibration = {joint: {"id": i + 1, "drive_mode": 0, "homing_offset": int(homing[joint]),
                               "range_min": int(mins[joint]), "range_max": int(maxes[joint])}
                       for i, joint in enumerate(JOINT_ORDER)}
        path.write_text(json.dumps(calibration, indent=4), encoding="utf-8")
        return path

    def disconnect(self) -> None:
        pass


class FakeTeleopArm:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self._t0 = time.monotonic()
        self._position = {joint: 0.0 for joint in JOINT_ORDER}

    def connect(self) -> None:
        self._t0 = time.monotonic()

    def read(self) -> dict[str, float]:
        if self.cfg.arm == "follower":
            return dict(self._position)
        t = time.monotonic() - self._t0
        wave = {joint: round(40.0 * math.sin(2 * math.pi * t / 4.0 + i), 2) for i, joint in enumerate(JOINT_ORDER)}
        wave["gripper"] = round(50.0 + 40.0 * math.sin(2 * math.pi * t / 3.0), 2)  # 0..100 like RANGE_0_100
        return wave

    def send(self, positions: dict[str, float]) -> dict[str, float]:
        cap = self.cfg.max_relative_target
        for joint, goal in positions.items():
            limit = cap.get(joint) if isinstance(cap, dict) else cap
            step = goal - self._position[joint]
            if limit is not None:
                step = max(-limit, min(limit, step))
            self._position[joint] = round(self._position[joint] + step, 2)
        return dict(self._position)

    def disconnect(self) -> None:
        pass


def fake_calibration_path(arm: str, arm_id: str) -> Path:
    return calibration_path(arm, arm_id).with_name(f"{arm_id}.fake.json")


class FakeBackend:
    name = "fake"

    def __init__(self, calibration_trajectory: Trajectory = default_trajectory):
        self.calibration_trajectory = calibration_trajectory
        self._lock = threading.Lock()
        self._unplug_at: dict[str, float] = {}

    def list_ports(self) -> list[str]:
        now = time.monotonic()
        with self._lock:
            gone = {FAKE_PORTS[arm] for arm, at in self._unplug_at.items() if now >= at}
        return sorted(p for p in [*FAKE_PORTS.values(), *_OTHER_PORTS] if p not in gone)

    def prepare_detection(self, arm: str) -> None:
        with self._lock:
            self._unplug_at[check_arm(arm)] = time.monotonic() + UNPLUG_AFTER_S

    def end_detection(self, arm: str) -> None:
        with self._lock:
            self._unplug_at.pop(arm, None)  # "plugged back in"

    def calibration_arm(self, cfg: ArmConfig) -> FakeCalibrationArm:
        return FakeCalibrationArm(cfg, self.calibration_trajectory)

    def teleop_arm(self, cfg: ArmConfig) -> FakeTeleopArm:
        return FakeTeleopArm(cfg)

    def calibration_file(self, arm: str, arm_id: str) -> Path:
        return fake_calibration_path(arm, arm_id)
