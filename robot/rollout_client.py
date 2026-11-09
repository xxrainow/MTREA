import argparse
from pathlib import Path

from robot.config import (
    FPS,
    POLICY_SERVER_HOST,
    POLICY_SERVER_PORT,
    RolloutGuardrails,
)

import math

from lab.core.hardware.cameras import load_camera_configs
from lab.core.hardware.devices import load_arm_config

from contextlib import suppress

from lab.core.hardware.devices import safe_disconnect
from lab.core.hardware.lock import HardwareLock
import logging


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        ) from None

    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        )

    return number

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one policy trial on the SO-101 follower."
    )
    parser.add_argument(
        "--server",
        default=f"{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument(
        "--max-relative-target",
        type=positive_float,
        required=True,
    )
    parser.add_argument(
        "--time-limit-s",
        type=positive_float,
        default=RolloutGuardrails().max_episode_seconds,
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser

def build_client_config(args: argparse.Namespace):
    if not args.checkpoint.strip():
        raise ValueError("checkpoint must not be empty")
    if not args.task.strip():
        raise ValueError("task must not be empty")
    if not math.isfinite(args.threshold):
        raise ValueError("threshold must be finite")

    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    arm = load_arm_config("follower")
    if not arm.use_degrees:
        raise ValueError("Rollouts require use_degrees=true")

    cameras = {
        camera.name: OpenCVCameraConfig(
            index_or_path=camera.index_or_path,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
        )
        for camera in load_camera_configs()
    }

    robot = SO101FollowerConfig(
        port=arm.port,
        id=arm.id,
        use_degrees=arm.use_degrees,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=False,
        cameras=cameras,
    )

    return RobotClientConfig(
        policy_type="pi05",
        pretrained_name_or_path=args.checkpoint,
        robot=robot,
        actions_per_chunk=args.actions_per_chunk,
        task=args.task,
        server_address=args.server,
        fps=args.fps,
        chunk_size_threshold=args.threshold,
        client_device="cpu",
    )

def create_locked_client(
    config,
    *,
    lock: HardwareLock,
    client_type,
):
    class ManagedClient(client_type):
        def __init__(self, config):
            try:
                super().__init__(config)
            except BaseException:
                channel = getattr(self, "channel", None)
                if channel is not None:
                    with suppress(Exception):
                        channel.close()

                safe_disconnect(
                    getattr(self, "robot", None),
                    "rollout client initialization",
                )
                raise

    lock.acquire("rollout")

    try:
        return ManagedClient(config)
    except BaseException:
        lock.release()
        raise


class PolicyConnectionError(RuntimeError):
    pass


class ClientErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def connect_policy(client) -> None:
    capture = ClientErrorCapture()
    client.logger.addHandler(capture)

    try:
        if not client.start():
            message = "\n".join(capture.messages)
            raise PolicyConnectionError(
                message or "The policy server refused or could not be reached"
            )
    finally:
        client.logger.removeHandler(capture)
