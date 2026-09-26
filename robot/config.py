"""Single source of truth for SO-101 Robot hardware constants.

Everything that needs an FPS, a joint name, a camera, or a network endpoint
imports it from here. Nothing else in the codebase defines these values, so
supporting another arm is a change to this file (and its siblings) rather than
a hunt through the tree. See AGENTS.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Hardware identity ---
ROBOT_TYPE = "so101_follower"
TELEOP_TYPE = "so101_leader"

# --- Control loop ---
FPS = 30

# Joint order. Must match the column order of `action` and `observation.state`
# in the recorded LeRobotDataset: reordering here silently corrupts every
# dataset and every checkpoint trained on one.
JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass(frozen=True)
class CameraConfig:
    name: str
    index_or_path: int | str  # OS camera index, or a device path / URL
    width: int
    height: int
    fps: int


CAMERAS = [
    CameraConfig(name="wrist", index_or_path=0, width=640, height=480, fps=FPS),
    CameraConfig(name="scene", index_or_path=2, width=640, height=480, fps=FPS),
]


def data_root() -> Path:
    """Root for datasets/checkpoints/logs/video (gitignored, never committed).

    Read from the ROBOLAB_DATA_ROOT env var so machines with different disk
    layouts (the A100 host vs a laptop) agree without touching code.
    """
    return Path(os.environ.get("ROBOLAB_DATA_ROOT", "~/robolab-data")).expanduser()


@dataclass(frozen=True)
class RolloutGuardrails:
    # If the action queue is starved this long, abort the rollout: a stalled
    # policy server must stop the arm, not let it coast on a stale command.
    queue_starved_ms: int = 500
    # Hard ceiling on one rollout so a policy that never terminates can't run
    # the arm indefinitely.
    max_episode_seconds: int = 60


# --- Policy server endpoint ---
# Bind to loopback only. Never 0.0.0.0: the policy server executes actions
# arriving over the network and must not be reachable off-host
# (AGENTS.md Boundaries, CVE-2026-25874). Reach the A100 host via SSH tunnel.
POLICY_SERVER_HOST = "127.0.0.1"
POLICY_SERVER_PORT = 8765

