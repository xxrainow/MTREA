"""Bounded, local IPC between the recorder child and the web service."""

import json
import math
import time
from contextlib import suppress
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

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
        self.phase_id = uuid4().hex
        atomic_write(self.root / "progress.json", json.dumps({
            "episode_phase": phase,
            "phase_id": self.phase_id,
            "phase_started": time.monotonic(),
            "phase_duration_s": duration,
            "current_episode": self.current,
            "saved_episodes": self.saved,
            "error": error,
        }, allow_nan=False).encode())


class FramePublisher:
    """Latest-frame mailbox: encoding never queues behind robot control."""

    def __init__(self, root: Path, *, max_frame_age_s: float | None = None,
                 write_metadata: bool = False):
        if max_frame_age_s is not None and (
            not math.isfinite(max_frame_age_s) or max_frame_age_s <= 0
        ):
            raise ValueError("max_frame_age_s must be a finite positive number")
        self.root = root
        self._max_frame_age_s = max_frame_age_s
        self._write_metadata = write_metadata
        self._lock = Lock()
        self._wake = Event()
        self._stop = Event()
        self._latest = None
        self._last_submit = None
        self._error = None
        self._worker = Thread(target=self._run, name="camera-frame-publisher", daemon=True)

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._worker.start()

    def submit(self, observation, *, captured_at: float | None = None):
        with self._lock:
            if self._stop.is_set():
                return
            now = time.monotonic()
            if self._last_submit is not None and now - self._last_submit < 0.2:
                return
            wall_time = time.time()
            captured_at = wall_time if captured_at is None else captured_at
            age = wall_time - captured_at
            if (
                not math.isfinite(age) or age < 0
                or self._max_frame_age_s is not None and age > self._max_frame_age_s
            ):
                return
            frames = {camera.name: observation[camera.name].copy()
                      for camera in CAMERAS if camera.name in observation}
            self._last_submit = now
            self._latest = (frames, now - age, captured_at)
        self._wake.set()

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def running(self):
        return self._worker.is_alive() and not self._stop.is_set()

    def close(self, timeout_s: float | None = None):
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout_s)
        if self._worker.is_alive():
            raise TimeoutError("Preview encoder did not stop")

    def _expired(self, captured: float) -> bool:
        return (
            self._max_frame_age_s is not None
            and time.monotonic() - captured > self._max_frame_age_s
        )

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                latest, self._latest = self._latest, None
            if latest is None:
                continue
            frames, captured, captured_at = latest
            if not frames or self._expired(captured):
                continue
            try:
                import cv2
                encoded = {}
                for name, frame in frames.items():
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok:
                        raise ValueError(f"Cannot encode {name} preview")
                    encoded[name] = jpeg.tobytes()
                if self._stop.is_set() or self._expired(captured):
                    continue
                for name, content in encoded.items():
                    atomic_write(self.root / f"{name}.jpg", content)
                if self._write_metadata:
                    atomic_write(self.root / "frames.json", json.dumps({
                        name: {"captured_at": captured_at,
                               "width": frame.shape[1], "height": frame.shape[0]}
                        for name, frame in frames.items()
                    }, allow_nan=False).encode())
                with self._lock:
                    self._error = None
                atomic_write(self.root / "preview_error.json", b"null")
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                with suppress(OSError):
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
            "phase_id": raw.get("phase_id"),
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
