import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab.core.collection import CollectionService
from lab.core.models import CollectionPhase, CollectionStatus
from lab.core.recorder import RecordingProcess
from lab.core.recording_telemetry import TelemetryReader, TelemetryWriter
from lab.core.recording_worker import recording_hooks
from lab.server.main import create_app


def recorder_module(stop_during=None):
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    calls = []
    robot = SimpleNamespace(get_observation=lambda: {"test": "observation"})

    class Dataset:
        def save_episode(self):
            calls.append("saved")

    def loop(robot, events, dataset=None, control_time_s=1, **kwargs):
        calls.append((dataset is not None, control_time_s))
        robot.get_observation()
        if stop_during == "preparation":
            events["stop_recording"] = True

    module = SimpleNamespace(
        record_loop=loop, make_robot_from_config=lambda _: robot,
        init_keyboard_listener=lambda: ("desktop-listener", {}), LeRobotDataset=Dataset,
    )
    return module, robot, events, calls


def test_hooks_prepare_record_reset_and_count_successful_saves(tmp_path, monkeypatch):
    monkeypatch.setattr("lab.core.recording_worker.PREPARATION_SECONDS", 0.02)
    module, robot, events, calls = recorder_module()
    original_loop = module.record_loop
    original_observation = robot.get_observation
    frames = []
    writer = TelemetryWriter(tmp_path)
    reader = TelemetryReader(tmp_path)
    with recording_hooks(module, writer, SimpleNamespace(submit=frames.append), events):
        assert module.init_keyboard_listener() == (None, events)
        module.make_robot_from_config(None)
        dataset = module.LeRobotDataset()
        module.record_loop(robot=robot, events=events, dataset=dataset, control_time_s=4)
        assert calls[:2] == [(False, 0.02), (True, 4)]
        assert reader.progress()["episode_phase"] == "saving"
        module.record_loop(robot=robot, events=events, control_time_s=2)
        assert reader.progress()["episode_phase"] == "resetting"
        assert reader.progress()["saved_episodes"] == 0
        dataset.save_episode()
        assert reader.progress()["saved_episodes"] == 1
        module.record_loop(robot=robot, events=events, dataset=dataset, control_time_s=4)
        assert reader.progress()["current_episode"] == 2
    assert module.record_loop is original_loop
    assert robot.get_observation is original_observation
    assert len(frames) == 4


def test_stop_during_preparation_never_enters_recording(tmp_path):
    module, robot, events, calls = recorder_module("preparation")
    writer = TelemetryWriter(tmp_path)
    with recording_hooks(module, writer, SimpleNamespace(submit=lambda _: None), events):
        with pytest.raises(KeyboardInterrupt):
            module.record_loop(robot=robot, events=events, dataset=module.LeRobotDataset(), control_time_s=4)
    assert len(calls) == 1
    assert writer.saved == 0


def test_hooks_keep_lerobot_image_writer_decorator(tmp_path):
    module, robot, events, calls = recorder_module()
    def safe_stop_image_writer(func):
        def wrapper(*args, **kwargs):
            calls.append("image-writer-wrapper")
            return func(*args, **kwargs)
        return wrapper
    module.record_loop = safe_stop_image_writer(module.record_loop)
    with recording_hooks(module, TelemetryWriter(tmp_path), SimpleNamespace(submit=lambda _: None), events):
        module.record_loop(robot=robot, events=events, dataset=module.LeRobotDataset(), control_time_s=4)
    assert calls.count("image-writer-wrapper") == 2


def test_hooks_keep_web_dataset_identity_without_timestamp_suffix(tmp_path):
    module, _, events, _ = recorder_module()
    class Config:
        repo_id = "local/demo"
        def stamp_repo_id(self):
            self.repo_id += "_timestamp"
    original = Config.stamp_repo_id
    module.DatasetRecordConfig = Config
    with recording_hooks(module, TelemetryWriter(tmp_path), SimpleNamespace(submit=lambda _: None), events):
        config = Config()
        config.stamp_repo_id()
        assert config.repo_id == "local/demo"
    assert Config.stamp_repo_id is original


def test_reader_timer_freshness_and_camera_path_restrictions(tmp_path):
    from robot.config import CAMERAS
    writer = TelemetryWriter(tmp_path)
    writer.stage("recording", 5)
    reader = TelemetryReader(tmp_path)
    assert reader.progress()["phase_duration_s"] == 5
    assert 0 <= reader.progress()["phase_elapsed_s"] <= 5
    name = CAMERAS[0].name
    frame = tmp_path / f"{name}.jpg"
    frame.write_bytes(b"jpeg")
    assert reader.frame(name) == b"jpeg"
    os.utime(frame, (1, 1))
    with pytest.raises(FileNotFoundError, match="stale"):
        reader.frame(name)
    with pytest.raises(FileNotFoundError):
        reader.frame("../../private")


def test_frame_publisher_uses_existing_observations_only(tmp_path, monkeypatch):
    from threading import Event
    from robot.config import CAMERAS
    from lab.core import recording_telemetry as telemetry
    ready = Event()
    original_write = telemetry.atomic_write

    def write(path, content):
        original_write(path, content)
        if path.suffix == ".jpg":
            ready.set()

    monkeypatch.setattr(telemetry, "atomic_write", write)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        COLOR_RGB2BGR=1, IMWRITE_JPEG_QUALITY=2,
        cvtColor=lambda frame, _: frame,
        imencode=lambda *_: (True, SimpleNamespace(tobytes=lambda: b"encoded-jpeg")),
    ))
    publisher = telemetry.FramePublisher(tmp_path)
    publisher.start()
    try:
        publisher.submit({CAMERAS[0].name: SimpleNamespace(copy=lambda: "existing-rgb-frame")})
        assert ready.wait(2)
        assert TelemetryReader(tmp_path).frame(CAMERAS[0].name) == b"encoded-jpeg"
    finally:
        publisher.close()


def test_frame_api_reads_recorder_without_opening_preview(tmp_path, monkeypatch):
    from robot.config import CAMERAS
    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    app = create_app(data_base=tmp_path)
    with TestClient(app) as client:
        service = CollectionService(data_base=tmp_path)
        service._fake_mode = False
        service._status = CollectionStatus(phase=CollectionPhase.RECORDING)
        service._recording = SimpleNamespace(recording_id="active-id", frame=lambda _: b"jpeg")
        app.state.collection = service
        response = client.get(f"/api/collection/frames/active-id/{CAMERAS[0].name}")
        assert response.status_code == 200
        assert response.content == b"jpeg"
        assert response.headers["cache-control"] == "no-store"
        assert client.get(f"/api/collection/frames/old-id/{CAMERAS[0].name}").status_code == 404
        service._status = CollectionStatus(phase=CollectionPhase.FINISHED)
        assert client.get(f"/api/collection/frames/active-id/{CAMERAS[0].name}").status_code == 404
        service._recording = None


def test_subprocess_ipc_waits_and_cleans_up(tmp_path, monkeypatch):
    # Use a harmless child to exercise the same process/IPC path without hardware.
    child = tmp_path / "child.py"
    child.write_text(
        "from pathlib import Path\n"
        "from lab.core.recording_telemetry import TelemetryWriter\n"
        "import sys\n"
        "writer=TelemetryWriter(Path(sys.argv[1]))\n"
        "writer.saved=2\n"
        "writer.current=2\n"
        "writer.stage('completed')\n"
    )
    import lab.core.recorder as recorder
    original_popen = recorder.subprocess.Popen

    def popen(command, **options):
        assert command[1:3] == ["-m", "lab.core.recording_worker"]
        assert options["stdin"] == recorder.subprocess.DEVNULL
        return original_popen([sys.executable, str(child), command[3]], **options)

    # The child is outside the checkout, so make the package import explicit.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setattr(recorder.subprocess, "Popen", popen)
    process = RecordingProcess(["lerobot-record"], telemetry=True)
    root = Path(process._telemetry_dir.name)
    try:
        assert process.wait() == 0
        assert process.progress()["saved_episodes"] == 2
        assert process.recording_id
    finally:
        process.close()
    assert not root.exists()


def test_worker_normalizes_options_and_reports_calibration_prompt(tmp_path, monkeypatch):
    import json
    from types import ModuleType
    from lab.core import recording_worker as worker
    module, _, _, _ = recorder_module()
    module.DatasetRecordConfig = SimpleNamespace(__dataclass_fields__={})
    closed = []
    class Publisher:
        def __init__(self, _): pass
        def start(self): pass
        def close(self): closed.append(True)
    def main():
        assert "--dataset.no_stamp=true" not in sys.argv
        assert "--display_data=false" in sys.argv
        assert "--play_sounds=false" in sys.argv
        raise EOFError("calibration prompt")
    module.main = main
    package = ModuleType("lerobot")
    scripts = ModuleType("lerobot.scripts")
    scripts.lerobot_record = module
    monkeypatch.setitem(sys.modules, "lerobot", package)
    monkeypatch.setitem(sys.modules, "lerobot.scripts", scripts)
    monkeypatch.setattr(worker.metadata, "version", lambda _: "0.6.0")
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(worker, "FramePublisher", Publisher)
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), "--dataset.no_stamp=true"])
    with pytest.raises(RuntimeError, match="calibration in Devices"):
        worker.main()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["episode_phase"] == "failed"
    assert "terminal input" in progress["error"]
    assert closed == [True]
