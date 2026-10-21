"""Bounded, local IPC between the recorder child and the web service."""

import json
import time
from pathlib import Path
from threading import Event, Lock, Thread

from robot.config import CAMERAS


PREPARATION_SECONDS = 3.0


def atomic_write(path: Path, content: bytes) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(content)
    pending.replace(path)


class TelemetryWriter:
    def __init__(self, root: Path):
        self.root = root
        self.saved = 0
        self.current = 1
        self.stage("connecting")

    def stage(self, phase: str, duration: float | None = None, error: str | None = None):
        atomic_write(self.root / "progress.json", json.dumps({
            "episode_phase": phase,
            "phase_started": time.monotonic(),
            "phase_duration_s": duration,
            "current_episode": self.current,
            "saved_episodes": self.saved,
            "error": error,
        }, allow_nan=False).encode())


class FramePublisher:
    """Latest-frame mailbox: encoding never queues behind robot control."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = Lock()
        self._wake = Event()
        self._stop = Event()
        self._latest = None
        self._last_submit = 0.0
        self._worker = Thread(target=self._run, daemon=True)

    def start(self):
        self._worker.start()

    def submit(self, observation):
        now = time.monotonic()
        if now - self._last_submit < 0.2:
            return
        self._last_submit = now
        frames = {camera.name: observation[camera.name].copy()
                  for camera in CAMERAS if camera.name in observation}
        with self._lock:
            self._latest = frames
        self._wake.set()

    def close(self):
        self._stop.set()
        self._wake.set()
        self._worker.join()

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                frames, self._latest = self._latest, None
            if not frames:
                continue
            try:
                import cv2
                for name, frame in frames.items():
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok:
                        raise ValueError(f"Cannot encode {name} preview")
                    atomic_write(self.root / f"{name}.jpg", jpeg.tobytes())
                atomic_write(self.root / "preview_error.json", b"null")
            except Exception as exc:
                atomic_write(self.root / "preview_error.json", json.dumps(str(exc)).encode())


class TelemetryReader:
    def __init__(self, root: Path):
        self.root = root
        self._last = {}

    def progress(self) -> dict:
        try:
            raw = json.loads((self.root / "progress.json").read_text())
            self._last = raw
        except (OSError, ValueError):
            raw = self._last
        if not raw:
            return {}
        duration = raw["phase_duration_s"]
        elapsed = max(0.0, time.monotonic() - raw["phase_started"])
        try:
            preview_error = json.loads((self.root / "preview_error.json").read_text())
        except (OSError, ValueError):
            preview_error = None
        return {
            "episode_phase": raw["episode_phase"],
            "phase_duration_s": duration,
            "phase_elapsed_s": min(duration, elapsed) if duration is not None else 0.0,
            "current_episode": raw["current_episode"],
            "saved_episodes": raw["saved_episodes"],
            "error": raw.get("error"),
            "preview_error": preview_error,
        }

    def frame(self, name: str) -> bytes:
        if name not in {camera.name for camera in CAMERAS}:
            raise FileNotFoundError("Unknown recording camera")
        path = self.root / f"{name}.jpg"
        if time.time() - path.stat().st_mtime > 2:
            raise FileNotFoundError("Recording frame is stale")
        return path.read_bytes()
