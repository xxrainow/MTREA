import json
import math
import time
from pathlib import Path

from lab.core.recording_telemetry import FramePublisher
from robot.config import CAMERAS


class TrialPreview:
    """Publish existing observations; this object never opens cameras."""

    def __init__(self, trial_dir: Path):
        self.root = trial_dir.resolve() / "frames"
        self._publisher = FramePublisher(
            self.root, max_frame_age_s=2.0, write_metadata=True
        )
        self._started = False
        self._closed = False
        self._error = None

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("A closed preview cannot be restarted")
        if not self._started:
            self._publisher.start()
            self._started = True

    def submit(self, observation: dict, *, captured_at: float | None = None) -> None:
        if self._closed:
            return
        try:
            self._publisher.submit(observation, captured_at=captured_at)
            self._error = None
        except Exception as exc:
            self._error = str(exc)

    def close(self) -> None:
        if self._started and not self._closed:
            try:
                self._publisher.close(timeout_s=5.0)
            except Exception as exc:
                self._error = str(exc)
            finally:
                self._closed = True

    def _frames(self) -> dict:
        try:
            frames = json.loads((self.root / "frames.json").read_text())
        except (OSError, ValueError):
            return {}
        return frames if isinstance(frames, dict) else {}

    @staticmethod
    def _captured_at(frame):
        captured = frame.get("captured_at") if isinstance(frame, dict) else None
        if (
            isinstance(captured, (int, float))
            and not isinstance(captured, bool)
            and math.isfinite(captured)
        ):
            return captured
        return None

    @classmethod
    def _fresh(cls, frame) -> bool:
        captured = cls._captured_at(frame)
        return captured is not None and 0 <= time.time() - captured <= 2.0

    def status(self) -> dict:
        frames = self._frames()
        error = self._error or self._publisher.error
        active = self._publisher.running
        return {
            "preview_active": active,
            "preview_error": error,
            "frames": {
                camera.name: {
                    "available": bool(
                        active and not error and self._fresh(frames.get(camera.name))
                        and (self.root / f"{camera.name}.jpg").is_file()
                    ),
                    "captured_at": self._captured_at(frames.get(camera.name)),
                }
                for camera in CAMERAS
            },
        }

    def frame(self, name: str) -> bytes:
        if name not in {camera.name for camera in CAMERAS}:
            raise FileNotFoundError("Unknown rollout camera")
        if not self.status()["frames"][name]["available"]:
            raise FileNotFoundError("Rollout frame is unavailable or stale")
        return (self.root / f"{name}.jpg").read_bytes()
