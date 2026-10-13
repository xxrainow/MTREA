"""Camera settings for preview, using the same YAML as recording.Read and inspect camera settings.
"""

from dataclasses import replace
from threading import Lock

import yaml

from lab.core.hardware import HardwareError
from lab.core.hardware.devices import CONFIGS_DIR
from robot.config import CAMERAS, CameraConfig


def load_camera_configs() -> list[CameraConfig]:
    path = CONFIGS_DIR / "cameras.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardwareError(f"cannot read {path}: {exc}") from None

    if not isinstance(raw, dict):
        raise HardwareError(f"{path} must contain a mapping")

    configs = []
    for default in CAMERAS:
        settings = raw.get(default.name)
        if not isinstance(settings, dict):
            raise HardwareError(f"{path}: missing camera {default.name}")

        source = settings.get("index_or_path")
        if (
            isinstance(source, bool)
            or not isinstance(source, (int, str))
            or isinstance(source, int) and source < 0
            or isinstance(source, str) and not source.strip()
        ):
            raise HardwareError(
                f"{default.name}: invalid index_or_path"
            )

        dimensions = {}
        for key in ("width", "height", "fps"):
            value = settings.get(key, getattr(default, key))
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise HardwareError(
                    f"{default.name}: {key} must be a positive integer"
                )
            dimensions[key] = value

        configs.append(
            replace(default, index_or_path=source, **dimensions)
        )

    return configs

# Manage opening, image loading, and closing of a single camera
# OpenCV function
class CameraPreview:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._capture = None
        self._lock = Lock()

    def start(self) -> None:
        import cv2

        with self._lock:
            if self._capture is not None:
                return

            capture = cv2.VideoCapture(self.config.index_or_path)
            try:
                if not capture.isOpened():
                    raise HardwareError(
                        f"{self.config.name}: cannot open camera "
                        f"{self.config.index_or_path}"
                    )

                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                capture.set(cv2.CAP_PROP_FPS, self.config.fps)

                ok, frame = capture.read()
                if not ok or frame is None:
                    raise HardwareError(
                        f"{self.config.name}: cannot read first frame"
                    )
            except Exception:
                capture.release()
                raise

            self._capture = capture

    def read_jpeg(self) -> bytes:
        import cv2

        with self._lock:
            if self._capture is None:
                raise HardwareError(
                    f"{self.config.name}: preview is not started"
                )

            try:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise HardwareError(
                        f"{self.config.name}: cannot read frame"
                    )

                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise HardwareError(
                        f"{self.config.name}: cannot encode frame"
                    )

                return encoded.tobytes()
            except cv2.error as exc:
                raise HardwareError(
                    f"{self.config.name}: camera read failed: {exc}"
                ) from None

    def stop(self) -> None:
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                finally:
                    self._capture = None


class CameraService:
    def __init__(self):
        self._previews: dict[str, CameraPreview] = {}
        self._error: str | None = None
        self._lock = Lock()

    def config(self) -> list[dict]:
        return [
            {
                "name": config.name,
                "width": config.width,
                "height": config.height,
                "fps": config.fps,
            }
            for config in load_camera_configs()
        ]

    def _status_locked(self) -> dict:
        return {
            "running": bool(self._previews),
            "cameras": list(self._previews),
            "error": self._error,
        }
        
    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def start(self) -> dict:
        with self._lock:
            if not self._previews:
                previews = [
                    CameraPreview(config)
                    for config in load_camera_configs()
                ]
                try:
                    for preview in previews:
                        preview.start()
                except Exception as exc:
                    for preview in previews:
                        preview.stop()
                        
                    self._error = str(exc)
                    raise HardwareError(self._error) from None

                self._previews = {
                    preview.config.name: preview
                    for preview in previews
                }
                self._error = None

            return self._status_locked()

    def _stop_locked(self) -> None:
        try:
            for preview in self._previews.values():
                preview.stop()
        finally:
            self._previews = {}

    def read_jpeg(self, name: str) -> bytes:
        with self._lock:
            preview = self._previews.get(name)

        if preview is None:
            raise HardwareError(f"{name}: preview is not started")

        try:
            return preview.read_jpeg()
        except HardwareError as exc:
            with self._lock:
                # An old stream must not stop a newly started session.
                if self._previews.get(name) is preview:
                    self._error = str(exc)
                    self._stop_locked()
            raise

    def stop(self) -> dict:
        with self._lock:
            self._stop_locked()
            self._error = None
            return self._status_locked()
