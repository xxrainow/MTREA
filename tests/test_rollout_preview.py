import json
import sys
import time
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from lab.core import recording_telemetry as telemetry
from robot.config import CAMERAS
from robot.rollout_preview import TrialPreview


def signal_frames_written(monkeypatch, ready):
    original = telemetry.atomic_write

    def write(path, content):
        original(path, content)
        if path.name == "frames.json":
            ready.set()

    monkeypatch.setattr(telemetry, "atomic_write", write)


def test_existing_rgb_observations_are_copied_and_encoded_off_thread(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    ready = Event()
    signal_frames_written(monkeypatch, ready)
    encoder_threads = []
    original_encode = cv2.imencode

    def encode(*args):
        encoder_threads.append(get_ident())
        return original_encode(*args)

    def forbidden_camera(*args, **kwargs):
        pytest.fail("Preview must not open a camera")

    monkeypatch.setattr(cv2, "imencode", encode)
    monkeypatch.setattr(cv2, "VideoCapture", forbidden_camera)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    preview = TrialPreview(tmp_path)
    preview.start()
    try:
        preview.submit({camera.name: image for camera in CAMERAS})
        image[:] = 0
        assert ready.wait(2)
        assert encoder_threads and all(thread != get_ident() for thread in encoder_threads)
        for camera in CAMERAS:
            jpeg = preview.frame(camera.name)
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (24, 32, 3)
            assert decoded[12, 16, 2] > 240
            assert decoded[12, 16, 0] < 15
        metadata = json.loads((preview.root / "frames.json").read_text())
        assert metadata[CAMERAS[0].name]["width"] == 32
        assert metadata[CAMERAS[0].name]["height"] == 24
    finally:
        preview.close()
    assert not preview.status()["preview_active"]
    with pytest.raises(FileNotFoundError):
        preview.frame(CAMERAS[0].name)


class Frame:
    shape = (2, 2, 3)

    def __init__(self, label):
        self.label = label

    def copy(self):
        return Frame(self.label)


@pytest.mark.parametrize("expire_first", [False, True])
def test_encoder_keeps_only_latest_pending_frame_and_drops_expired_frames(
    tmp_path, monkeypatch, expire_first
):
    now = [0.0]
    entered, resume, ready = Event(), Event(), Event()
    calls, writes = [], []
    original = telemetry.atomic_write

    def encode(_extension, frame, _options):
        calls.append(frame.label)
        if frame.label == "first":
            entered.set()
            assert resume.wait(2)
        return True, SimpleNamespace(tobytes=lambda: frame.label.encode())

    def write(path, content):
        original(path, content)
        if path.suffix == ".jpg":
            writes.append(content)
        if path.name == "frames.json" and writes[-1] == b"latest":
            ready.set()

    monkeypatch.setattr(telemetry.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(telemetry, "atomic_write", write)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        COLOR_RGB2BGR=1, IMWRITE_JPEG_QUALITY=2,
        cvtColor=lambda frame, _: frame, imencode=encode,
    ))
    preview = TrialPreview(tmp_path)
    preview.start()
    name = CAMERAS[0].name
    try:
        preview.submit({name: Frame("first")})
        assert entered.wait(2)
        now[0] = 0.25
        preview.submit({name: Frame("discarded")})
        now[0] = 3.0 if expire_first else 0.5
        preview.submit({name: Frame("latest")})
        resume.set()
        assert ready.wait(2)
        assert calls == ["first", "latest"]
        assert writes == ([b"latest"] if expire_first else [b"first", b"latest"])
    finally:
        resume.set()
        preview.close()


@pytest.mark.parametrize("captured", [1.0, float("nan"), "bad", True])
def test_recent_file_mtime_cannot_make_stale_or_invalid_capture_fresh(
    tmp_path, captured
):
    preview = TrialPreview(tmp_path)
    preview.start()
    name = CAMERAS[0].name
    try:
        (preview.root / f"{name}.jpg").write_bytes(b"old-image")
        (preview.root / "frames.json").write_text(json.dumps({
            name: {"captured_at": captured},
        }))
        assert not preview.status()["frames"][name]["available"]
        with pytest.raises(FileNotFoundError):
            preview.frame(name)
        json.dumps(preview.status(), allow_nan=False)
        with pytest.raises(FileNotFoundError, match="Unknown"):
            preview.frame("../../session")
    finally:
        preview.close()


def test_encoding_failure_is_exposed_and_worker_can_recover(tmp_path, monkeypatch):
    error_written, ready = Event(), Event()
    failed = [True]
    original = telemetry.atomic_write

    def encode(*args):
        return (False, None) if failed[0] else (
            True, SimpleNamespace(tobytes=lambda: b"recovered")
        )

    def write(path, content):
        original(path, content)
        if path.name == "preview_error.json":
            (ready if content == b"null" else error_written).set()

    monkeypatch.setattr(telemetry, "atomic_write", write)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        COLOR_RGB2BGR=1, IMWRITE_JPEG_QUALITY=2,
        cvtColor=lambda frame, _: frame, imencode=encode,
    ))
    preview = TrialPreview(tmp_path)
    preview.start()
    name = CAMERAS[0].name
    try:
        preview.submit({name: Frame("bad")})
        assert error_written.wait(2)
        assert "Cannot encode" in preview.status()["preview_error"]
        failed[0] = False
        # Advance only the publisher's rate limiter, without sleeping in the test.
        with preview._publisher._lock:
            preview._publisher._last_submit = None
        preview.submit({name: Frame("good")})
        assert ready.wait(2)
        assert preview.frame(name) == b"recovered"
        assert preview.status()["preview_error"] is None
    finally:
        preview.close()


def test_observation_already_stale_at_submit_is_not_encoded(tmp_path):
    preview = TrialPreview(tmp_path)
    preview.start()
    try:
        preview.submit(
            {CAMERAS[0].name: Frame("old")}, captured_at=time.time() - 3.0,
        )
    finally:
        preview.close()
    assert not (preview.root / "frames.json").exists()
    assert not list(preview.root.glob("*.jpg"))
