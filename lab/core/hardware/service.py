"""Everything the server and the CLI need, behind one object per process.

Holds the backend, the hardware lock, the joint stream and the sessions.
Methods raise HardwareError subclasses with messages meant for the operator;
lab/server/hardware_api.py maps them to HTTP status codes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from lab.core import paths
from lab.core.hardware import ARMS, HardwareBusy, HardwareError, JointStream, check_arm
from lab.core.hardware.calibration import CalibrationSession
from lab.core.hardware.devices import (
    CONFIGS_DIR, RealBackend, config_path, legacy_calibration_path, load_arm_config,
)
from lab.core.hardware.lock import HardwareLock
from lab.core.hardware.ports import PortDetection, save_port
from lab.core.hardware.teleop import TeleopSession

BACKEND_ENV = "MTREA_HARDWARE"


def make_backend() -> Any:
    choice = os.environ.get(BACKEND_ENV, "real").strip().lower() or "real"
    if choice == "real":
        return RealBackend()
    if choice == "fake":
        from lab.core.hardware.fake import FakeBackend

        return FakeBackend()
    raise HardwareError(f"{BACKEND_ENV} must be 'real' or 'fake', got {choice!r}")


def fake_configs_dir(data_base: Path | None = None) -> Path:
    """Fake mode edits a copy of the arm yamls under data/fake/, never robot/configs/."""
    target = paths.root(data_base) / "fake" / "robot_configs"
    target.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        if not config_path(arm, target).exists():
            shutil.copyfile(config_path(arm, CONFIGS_DIR), config_path(arm, target))
    return target


class HardwareService:
    def __init__(self, data_base: Path | None = None, backend: Any = None, configs_dir: Path | None = None):
        self.backend = backend if backend is not None else make_backend()
        if configs_dir is None:
            configs_dir = fake_configs_dir(data_base) if self.backend.name == "fake" else CONFIGS_DIR
        self.configs_dir = configs_dir
        self.lock = HardwareLock(paths.hardware_lock(data_base))
        self.stream = JointStream()
        self.calibration = CalibrationSession(self.backend, self.lock, self.stream, configs_dir)
        self.teleop = TeleopSession(self.backend, self.lock, self.stream, configs_dir)
        self._detections = {arm: PortDetection(lister=self.backend.list_ports) for arm in ARMS}

    # --- config and ports ---

    def config(self) -> dict:
        ports = self._ports_or_none()
        arms = {}
        for arm in ARMS:
            try:
                cfg = load_arm_config(arm, self.configs_dir)
            except HardwareError as exc:
                arms[arm] = {"error": str(exc), "config_file": str(config_path(arm, self.configs_dir))}
                continue
            calibration = self.backend.calibration_file(arm, cfg.id)
            legacy = legacy_calibration_path(arm, cfg.id)
            arms[arm] = {
                "config_file": str(config_path(arm, self.configs_dir)),
                "port": cfg.port,
                "port_present": None if ports is None else cfg.port in ports,
                "id": cfg.id,
                "use_degrees": cfg.use_degrees,
                "max_relative_target": cfg.max_relative_target,
                "calibration_file": str(calibration),
                "calibration_present": calibration.is_file(),
                # Written by lerobot < 0.6; v0.6.1 ignores it and looks at calibration_file.
                "legacy_calibration_present": legacy.is_file(),
            }
        return {"backend": self.backend.name, "arms": arms, "ports": ports, "lock": self.lock.holder()}

    def _ports_or_none(self) -> list[str] | None:
        try:
            return self.backend.list_ports()
        except HardwareError:
            return None

    def ports(self) -> dict:
        return {"backend": self.backend.name, "ports": self.backend.list_ports()}

    def detect_start(self, arm: str) -> dict:
        check_arm(arm)
        holder = self.lock.holder()
        if holder is not None:
            raise HardwareBusy(f"the arms are in use by {holder.get('owner', 'another session')}; "
                               "stop it before unplugging an arm")
        ports = self._detections[arm].start()
        self.backend.prepare_detection(arm)
        return {"backend": self.backend.name, "arm": arm, "ports_before": ports,
                "instruction": f"Unplug the USB cable of the {arm} arm now."}

    def detect_finish(self, arm: str, timeout_s: float | None = None) -> dict:
        check_arm(arm)
        try:
            kwargs = {} if timeout_s is None else {"timeout_s": timeout_s}
            port = self._detections[arm].finish(**kwargs)
        finally:
            self.backend.end_detection(arm)
        return {"backend": self.backend.name, "arm": arm, "port": port,
                "instruction": f"Plug the {arm} arm back in. Save {port} to use it."}

    def save_port(self, arm: str, port: str) -> dict:
        path = save_port(arm, port, self.configs_dir)
        return {"backend": self.backend.name, "arm": arm, "port": port, "config_file": str(path)}

    def shutdown(self) -> None:
        for session in (self.teleop, self.calibration):
            try:
                session.shutdown()
            except HardwareError:
                pass
