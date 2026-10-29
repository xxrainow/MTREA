import json
from pathlib import Path
from threading import Lock

import yaml
import os
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4
from dataclasses import dataclass
from datetime import datetime

from lab.core.dataset_metadata import verify
from lab.core.hardware import HardwareError, SessionStateError
from lab.core.hardware.devices import CONFIGS_DIR
from robot.config import CAMERAS, FPS, ROBOT_TYPE, TELEOP_TYPE


def build_record_command(
    *,
    executable: str,
    hardware_args: list[str],
    repo_id: str,
    dataset_root: Path,
    instruction: str,
    num_episodes: int,
    episode_time_s: float,
    reset_time_s: float,
    resume: bool,
) -> list[str]:
    return [
        executable,
        *hardware_args,
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={instruction}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.fps={FPS}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        "--dataset.push_to_hub=false",
        "--dataset.no_stamp=true",
        f"--resume={'true' if resume else 'false'}",
    ]

def _load_recording_config(
    configs_dir: Path,
    filename: str,
) -> dict:
    path = configs_dir / filename
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardwareError(f"Cannot read {path}: {exc}") from None

    if not isinstance(raw, dict):
        raise HardwareError(f"{path}: expected a mapping")

    return raw


def build_hardware_args(
    configs_dir: Path = CONFIGS_DIR,
) -> list[str]:
    follower = _load_recording_config(
        configs_dir, "so101_follower.yaml"
    )
    leader = _load_recording_config(
        configs_dir, "so101_leader.yaml"
    )
    camera_settings = _load_recording_config(
        configs_dir, "cameras.yaml"
    )

    for name, arm in (("follower", follower), ("leader", leader)):
        for key in ("port", "id"):
            value = arm.get(key)
            if not isinstance(value, str) or not value.strip():
                raise HardwareError(
                    f"{name}: {key} must be a non-empty string"
                )

        if not isinstance(arm.get("use_degrees"), bool):
            raise HardwareError(
                f"{name}: use_degrees must be true or false"
            )

    cameras = {}
    for camera in CAMERAS:
        settings = camera_settings.get(camera.name)
        if not isinstance(settings, dict):
            raise HardwareError(
                f"Missing camera settings: {camera.name}"
            )

        source = settings.get("index_or_path")
        if (
            isinstance(source, bool)
            or not isinstance(source, (int, str))
            or isinstance(source, int) and source < 0
            or isinstance(source, str) and not source.strip()
        ):
            raise HardwareError(
                f"{camera.name}: invalid index_or_path"
            )

        cameras[camera.name] = {
            "type": "opencv",
            "index_or_path": source,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
        }

    return [
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.port={follower['port']}",
        f"--robot.id={follower['id']}",
        f"--robot.use_degrees={str(follower['use_degrees']).lower()}",
        f"--robot.cameras={json.dumps(cameras)}",
        f"--teleop.type={TELEOP_TYPE}",
        f"--teleop.port={leader['port']}",
        f"--teleop.id={leader['id']}",
        f"--teleop.use_degrees={str(leader['use_degrees']).lower()}",
    ]

@dataclass(frozen=True)
class RecordingResult:
    started_at: str
    ended_at: str
    exit_code: int
    episodes_after: int | None
    problems: tuple[str, ...]


class RecordingProcess:
    def __init__(self, command: list[str], *, telemetry: bool = False) -> None:
        self.started_at = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        self.recording_id = uuid4().hex if telemetry else None
        self._telemetry_dir = TemporaryDirectory(prefix="mtrea-recording-") if telemetry else None
        self._telemetry = None
        self._command_lock = Lock()
        options = {}
        if telemetry:
            from lab.core.recording_telemetry import TelemetryReader
            self._telemetry = TelemetryReader(Path(self._telemetry_dir.name))
            command = [sys.executable, "-m", "lab.core.recording_worker", self._telemetry_dir.name, *command[1:]]
            options = {"cwd": Path(__file__).resolve().parents[2], "stdin": subprocess.DEVNULL}
        try:
            if os.name == "nt":
                self._process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **options)
            else:
                self._process = subprocess.Popen(command, start_new_session=True, **options)
        except BaseException:
            if self._telemetry_dir is not None:
                self._telemetry_dir.cleanup()
            raise

    def progress(self) -> dict:
        return self._telemetry.progress() if self._telemetry else {}

    def send_command(
        self,
        action: str,
        *,
        expected_phase: str,
        expected_episode: int,
    ) -> str:
        from lab.core.recording_telemetry import atomic_write

        allowed_actions = {
            "start_episode",
            "finish_episode",
            "retry_episode",
        }

        if action not in allowed_actions:
            raise ValueError(f"Unknown recording action: {action}")

        if expected_episode < 1:
            raise ValueError("Expected episode must be positive")

        with self._command_lock:
            if self._telemetry_dir is None:
                raise SessionStateError(
                    "Commands require a web recording process"
                )

            if self._process.poll() is not None:
                raise SessionStateError(
                    "Recording process has already exited"
                )

            root = Path(self._telemetry_dir.name)
            command_path = root / "command.json"

            if command_path.exists():
                raise SessionStateError(
                    "A recording command is still pending"
                )

            command_id = uuid4().hex
            payload = {
                "id": command_id,
                "recording_id": self.recording_id,
                "action": action,
                "expected_phase": expected_phase,
                "expected_episode": expected_episode,
            }

            atomic_write(
                command_path,
                json.dumps(
                    payload,
                    allow_nan=False,
                ).encode("utf-8"),
            )

            return command_id

    def frame(self, name: str) -> bytes:
        if self._telemetry is None:
            raise FileNotFoundError("Recording preview is unavailable")
        return self._telemetry.frame(name)

    def close(self):
        if self._telemetry_dir is not None and self._process.poll() is not None:
            self._telemetry_dir.cleanup()

    def wait(self) -> int:
        return self._process.wait()

    def request_stop(self) -> None:
        if self._process.poll() is not None:
            return

        try:
            if os.name == "nt":
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass


def wait_and_verify(
    recording: RecordingProcess,
    *,
    dataset_root: Path,
    episodes_before: int,
    requested_episodes: int,
    instruction: str,
) -> RecordingResult:
    exit_code = recording.wait()
    ended_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    episodes_after, problems = verify(
        dataset_root,
        episodes_before,
        requested_episodes,
        instruction,
        exit_code,
    )

    return RecordingResult(
        started_at=recording.started_at,
        ended_at=ended_at,
        exit_code=exit_code,
        episodes_after=episodes_after,
        problems=tuple(problems),
    )


