"""Test-only hardware replacement; keeps LeRobot's actual client and gRPC calls."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import serial

from lab.core.hardware.lock import HardwareLock
from robot.config import CAMERAS, FPS, JOINT_ORDER
from robot.rollout_client import create_locked_client
from robot.rollout_worker import run_process


def forbid_hardware(*args, **kwargs):
    raise AssertionError("Physical hardware access is forbidden in this test")


class FakeCamera:
    is_connected = True

    def disconnect(self):
        self.is_connected = False


class FakeBus:
    def __init__(self, positions):
        self.positions = positions
        self.is_connected = False
        self.torque_disabled = False
        self.hold_targets = []

    def sync_read(self, register, **kwargs):
        assert register == "Present_Position"
        return dict(self.positions)

    def sync_write(self, register, positions):
        assert register == "Goal_Position"
        self.hold_targets.append(dict(positions))

    def disable_torque(self, **kwargs):
        self.torque_disabled = True

    def disconnect(self, **kwargs):
        self.is_connected = False


class FakeRobot:
    def __init__(self, config):
        self.config = config
        self.initial = {joint: float(index + 10) for index, joint in enumerate(JOINT_ORDER)}
        self.bus = FakeBus(self.initial)
        self.cameras = {camera: FakeCamera() for camera in config.cameras}
        self.action_features = {f"{joint}.pos": float for joint in JOINT_ORDER}
        self.images = {
            name: np.full((camera.height, camera.width, 3), 90, dtype=np.uint8)
            for name, camera in config.cameras.items()
        }
        self.observation_features = {
            **self.action_features, **{name: image.shape for name, image in self.images.items()},
        }
        self.actions = []
        self.observations = 0

    def connect(self):
        self.bus.is_connected = True

    def get_observation(self):
        self.observations += 1
        return {**{f"{key}.pos": value for key, value in self.initial.items()}, **self.images}

    def send_action(self, action):
        self.actions.append(dict(action))
        return action


def create_fake_client(config, lock, trial_dir):
    previous = os.getcwd()
    try:
        os.chdir(trial_dir)
        from lerobot.async_inference import robot_client as module
    finally:
        os.chdir(previous)
    original = module.make_robot_from_config
    module.make_robot_from_config = FakeRobot
    try:
        client = create_locked_client(config, lock=lock, client_type=module.RobotClient)
    finally:
        module.make_robot_from_config = original

    release = client.release_arm

    def release_and_report():
        release()
        robot = client.robot
        (trial_dir / "fake_robot.json").write_text(json.dumps({
            "initial": robot.initial, "actions": robot.actions,
            "observations": robot.observations, "hold_targets": robot.bus.hold_targets,
            "torque_disabled": robot.bus.torque_disabled,
            "bus_connected": robot.bus.is_connected,
            "cameras_connected": any(camera.is_connected for camera in robot.cameras.values()),
        }))

    client.release_arm = release_and_report
    return client


def main():
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    trial, server, checkpoint, mode = sys.argv[1:]
    # Any accidental fallback to native device creation fails before opening I/O.
    cv2.VideoCapture = forbid_hardware
    serial.Serial = forbid_hardware
    cameras = {
        camera.name: OpenCVCameraConfig(
            index_or_path=-999, height=camera.height,
            width=camera.width + (1 if mode == "shape_mismatch" else 0), fps=FPS,
        ) for camera in CAMERAS
    }
    config = RobotClientConfig(
        policy_type="pi05", pretrained_name_or_path=checkpoint,
        robot=SO101FollowerConfig(port="FORBIDDEN", id="synthetic", cameras=cameras,
                                  use_degrees=True, max_relative_target=5.0,
                                  disable_torque_on_disconnect=False),
        server_address=server, task="Synthetic hold integration", fps=FPS,
        actions_per_chunk=50, client_device="cpu",
    )
    return run_process(config, trial_dir=Path(trial),
                       time_limit_s=1.5 if mode == "time_limit" else 30,
                       client_factory=create_fake_client,
                       lock=HardwareLock(Path(trial).parent / "test-hardware.lock"))


if __name__ == "__main__":
    raise SystemExit(main())
