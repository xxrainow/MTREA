"""Arm settings from robot/configs/so101_*.yaml, and the real (LeRobot) backend.

Robot constants come from robot/config.py; per-machine values (port, id,
use_degrees, max_relative_target) from the yaml. LeRobot is imported inside
the functions that need it, so nothing here requires it until an arm is
opened.

LeRobot calls below were checked against the v0.6.1 tag:
- robots/so_follower/so_follower.py, teleoperators/so_leader/so_leader.py:
  connect(calibrate=True) would call calibrate(), which blocks on input();
  connect_for_teleop() does the same steps without prompting.
- motors/motors_bus.py: set_half_turn_homings(), sync_read(normalize=False,
  num_retry=), write_calibration(), disconnect(disable_torque).
- robots/robot.py, teleoperators/teleoperator.py: the calibration file is
  HF_LEROBOT_CALIBRATION/{robots,teleoperators}/<class name>/<id>.json.
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lab.core.hardware import HardwareError, check_arm
from robot.config import JOINT_ORDER, ROBOT_TYPE, TELEOP_TYPE

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "robot" / "configs"
_TYPES = {"follower": ROBOT_TYPE, "leader": TELEOP_TYPE}

# SOFollower.name / SOLeader.name in v0.6.1. The older so101_follower/ and
# so101_leader/ directories are no longer read by lerobot.
_CALIBRATION_SUBDIRS = {"follower": ("robots", "so_follower"), "leader": ("teleoperators", "so_leader")}
_LEGACY_SUBDIRS = {"follower": ("robots", "so101_follower"), "leader": ("teleoperators", "so101_leader")}


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    type: str
    port: str
    id: str
    use_degrees: bool
    # Follower only; LeRobot's clamp per step, in the joint units (degrees with use_degrees).
    max_relative_target: float | dict[str, float] | None = None


def config_path(arm: str, configs_dir: Path = CONFIGS_DIR) -> Path:
    return configs_dir / f"so101_{check_arm(arm)}.yaml"


def load_arm_config(arm: str, configs_dir: Path = CONFIGS_DIR) -> ArmConfig:
    path = config_path(arm, configs_dir)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardwareError(f"cannot read {path}: {exc}") from None
    if not isinstance(raw, dict):
        raise HardwareError(f"{path} is not a mapping")
    if raw.get("type") != _TYPES[arm]:
        raise HardwareError(f"{path}: type is {raw.get('type')!r}, expected {_TYPES[arm]!r}")
    for key in ("port", "id"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise HardwareError(f"{path}: {key} must be a non-empty string")
    use_degrees = raw.get("use_degrees")
    if not isinstance(use_degrees, bool):
        raise HardwareError(f"{path}: use_degrees must be true or false")
    return ArmConfig(arm=arm, type=raw["type"], port=raw["port"], id=raw["id"], use_degrees=use_degrees,
                     max_relative_target=_max_relative_target(raw.get("max_relative_target"), path)
                     if arm == "follower" else None)


def _max_relative_target(value: Any, path: Path) -> float | dict[str, float] | None:
    # ensure_safe_goal_position() in v0.6.1 accepts float or dict only: an int raises TypeError.
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    if isinstance(value, dict) and set(value) == set(JOINT_ORDER) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in value.values()):
        return {k: float(v) for k, v in value.items()}
    raise HardwareError(f"{path}: max_relative_target must be null, a positive number, "
                        f"or a positive number for each of {JOINT_ORDER}")


# --- calibration files ---

def calibration_root() -> Path:
    """HF_LEROBOT_CALIBRATION as lerobot.utils.constants resolves it (with huggingface_hub's HF_HOME)."""
    if os.getenv("HF_LEROBOT_CALIBRATION"):
        return Path(os.environ["HF_LEROBOT_CALIBRATION"]).expanduser()
    if os.getenv("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"]).expanduser() / "calibration"
    hf_home = os.getenv("HF_HOME") or os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    return Path(os.path.expandvars(hf_home)).expanduser() / "lerobot" / "calibration"


def calibration_path(arm: str, arm_id: str) -> Path:
    return calibration_root().joinpath(*_CALIBRATION_SUBDIRS[check_arm(arm)], f"{arm_id}.json")


def legacy_calibration_path(arm: str, arm_id: str) -> Path:
    return calibration_root().joinpath(*_LEGACY_SUBDIRS[check_arm(arm)], f"{arm_id}.json")


# --- disconnect ---

def safe_disconnect(device: Any, context: str = "cleanup") -> None:
    """Disconnect a LeRobot device; if that fails, force the serial port closed.

    A failed disconnect (e.g. disabling torque on a flaky USB link) can leave
    the port handle open until the process exits, which on Windows blocks every
    later open of that COM port. Same fallback as LeLab's devices.py.
    """
    if device is None:
        return
    try:
        device.disconnect()
        return
    except Exception as exc:
        logger.warning("disconnect failed during %s: %s", context, exc)
    port_handler = getattr(getattr(device, "bus", None), "port_handler", None)
    if port_handler is not None:
        with suppress(Exception):
            port_handler.clearPort()
        with suppress(Exception):
            port_handler.is_using = False
        try:
            port_handler.closePort()
            logger.info("force-closed the serial port after a failed disconnect")
        except Exception as exc:
            logger.warning("could not force-close the serial port: %s", exc)
    for cam in (getattr(device, "cameras", None) or {}).values():
        with suppress(Exception):
            cam.disconnect()


# --- real backend ---

def make_device(cfg: ArmConfig) -> Any:
    """LeRobot follower/leader for cfg, without cameras.

    Calibration and teleoperation never read the cameras; lerobot-record
    opens them itself from robot/configs/cameras.yaml.
    """
    try:
        if cfg.arm == "follower":
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        else:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    except ImportError as exc:
        raise HardwareError(f"lerobot is not importable from this Python ({exc}); activate the lerobot "
                            "environment, or set MTREA_HARDWARE=fake to work without arms") from None
    if cfg.arm == "follower":
        return SO101Follower(SO101FollowerConfig(
            port=cfg.port, id=cfg.id, use_degrees=cfg.use_degrees,
            max_relative_target=cfg.max_relative_target))
    return SO101Leader(SO101LeaderConfig(port=cfg.port, id=cfg.id, use_degrees=cfg.use_degrees))


def _open_bus(device: Any, cfg: ArmConfig) -> None:
    try:
        device.bus.connect()
    except Exception as exc:
        raise HardwareError(f"could not open the {cfg.arm} arm on {cfg.port}: {exc}. "
                            "Check that it is plugged in and powered, or detect its port again.") from None


class RealCalibrationArm:
    """The steps of SOFollower/SOLeader.calibrate() (v0.6.1), without input()."""

    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.device: Any = None

    def connect(self) -> None:
        self.device = make_device(self.cfg)
        from lerobot.motors.feetech import OperatingMode

        _open_bus(self.device, self.cfg)
        bus = self.device.bus
        bus.disable_torque()  # the operator moves the arm by hand
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def read_raw(self) -> dict[str, int]:
        return dict(self.device.bus.sync_read("Present_Position", normalize=False, num_retry=5))

    def set_half_turn_homings(self) -> dict[str, int]:
        return dict(self.device.bus.set_half_turn_homings())

    def save(self, homing: dict[str, int], mins: dict[str, int], maxes: dict[str, int]) -> Path:
        from lerobot.motors import MotorCalibration

        bus = self.device.bus
        calibration = {
            motor: MotorCalibration(id=m.id, drive_mode=0, homing_offset=int(homing[motor]),
                                    range_min=int(mins[motor]), range_max=int(maxes[motor]))
            for motor, m in bus.motors.items()
        }
        bus.write_calibration(calibration)
        self.device.calibration = calibration
        self.device._save_calibration()  # lerobot's own writer, to the path lerobot reads
        return Path(self.device.calibration_fpath)

    def disconnect(self) -> None:
        safe_disconnect(self.device, f"{self.cfg.arm} calibration")
        self.device = None


class RealTeleopArm:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.device: Any = None

    def connect(self) -> None:
        device = make_device(self.cfg)
        _open_bus(device, self.cfg)
        try:
            if not device.calibration:
                raise HardwareError(f"no calibration for the {self.cfg.arm} arm (id {self.cfg.id}) at "
                                    f"{device.calibration_fpath}; calibrate it first")
            # connect() would prompt on stdin here; pressing Enter there does exactly this.
            if not device.is_calibrated:
                device.bus.write_calibration(device.calibration)
            device.configure()
        except BaseException:
            safe_disconnect(device, f"{self.cfg.arm} connect")
            raise
        self.device = device

    def read(self) -> dict[str, float]:
        raw = self.device.get_observation() if self.cfg.arm == "follower" else self.device.get_action()
        return {key.removesuffix(".pos"): float(v) for key, v in raw.items() if key.endswith(".pos")}

    def send(self, positions: dict[str, float]) -> dict[str, float]:
        """Follower only: command positions; returns what was sent after max_relative_target clamping."""
        sent = self.device.send_action({f"{joint}.pos": v for joint, v in positions.items()})
        return {key.removesuffix(".pos"): float(v) for key, v in sent.items()}

    def disconnect(self) -> None:
        safe_disconnect(self.device, f"{self.cfg.arm} teleop")
        self.device = None


class RealBackend:
    name = "real"

    def list_ports(self) -> list[str]:
        from lab.core.hardware.ports import list_ports

        return list_ports()

    def prepare_detection(self, arm: str) -> None:
        pass  # the operator unplugs the cable

    def end_detection(self, arm: str) -> None:
        pass

    def calibration_arm(self, cfg: ArmConfig) -> RealCalibrationArm:
        return RealCalibrationArm(cfg)

    def teleop_arm(self, cfg: ArmConfig) -> RealTeleopArm:
        return RealTeleopArm(cfg)

    def calibration_file(self, arm: str, arm_id: str) -> Path:
        return calibration_path(arm, arm_id)
