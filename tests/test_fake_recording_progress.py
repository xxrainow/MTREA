import json
from threading import Event, Thread
from time import monotonic, sleep

from fastapi.testclient import TestClient

from lab.core import collection as collection_module
from lab.core.collection import CollectionService
from lab.core.collection_input import prepare_start_arguments
from lab.core.collection_ledger import read_ledger
from lab.core.fake_recorder import FakeRecordingProcess
from lab.core.models import CollectionPhase
from lab.core.recorder import build_record_command
from lab.server.main import create_app


def test_fake_honors_phase_durations_and_does_not_save_twice(tmp_path):
    stages = []

    class ObservedRecording(FakeRecordingProcess):
        PREPARATION_SECONDS = 0.01

        def _stage(self, phase, duration=0.0):
            stages.append((phase, duration))
            return super()._stage(phase, duration)

    command = build_record_command(
        executable="unused",
        hardware_args=[],
        repo_id="local/demo",
        dataset_root=tmp_path / "demo",
        instruction="Move the block.",
        num_episodes=2,
        episode_time_s=0.02,
        reset_time_s=0.03,
        resume=False,
    )
    recording = ObservedRecording(command)
    assert recording.wait() == 0
    assert recording.wait() == 0
    assert stages == [
        ("preparing", 0.01), ("recording", 0.02), ("resetting", 0.03),
        ("saving", 0.0), ("recording", 0.02), ("saving", 0.0),
        ("completed", 0.0),
    ]
    progress = recording.progress()
    assert progress["saved_episodes"] == 2
    assert progress["current_episode"] == 2
    assert json.loads((tmp_path / "demo/meta/info.json").read_text())["total_episodes"] == 2


def test_stop_interrupts_preparation_without_saving_episode(tmp_path):
    preparing = Event()

    class ObservedRecording(FakeRecordingProcess):
        PREPARATION_SECONDS = 30

        def _stage(self, phase, duration=0.0):
            if phase == "preparing":
                preparing.set()
            return super()._stage(phase, duration)

    recording = ObservedRecording(build_record_command(
        executable="unused", hardware_args=[], repo_id="local/demo",
        dataset_root=tmp_path / "demo", instruction="Move the block.",
        num_episodes=2, episode_time_s=30, reset_time_s=15, resume=False,
    ))
    worker = Thread(target=recording.wait)
    worker.start()
    try:
        assert preparing.wait(2)
    finally:
        recording.request_stop()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert recording.progress()["episode_phase"] == "stopped"
    assert recording.progress()["saved_episodes"] == 0


def request():
    return {
        "dataset": "progress_demo", "seed": 42, "session_index": 1,
        "sessions": [{"instruction": "Move the block.", "num_episodes": 2,
                      "episode_time_s": 0.03, "reset_time_s": 0.03}],
    }


def test_service_keeps_final_fake_progress_and_saved_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    monkeypatch.setattr(FakeRecordingProcess, "PREPARATION_SECONDS", 0.01)
    arguments = prepare_start_arguments(request(), data_base=tmp_path)
    service = CollectionService(data_base=tmp_path)
    service.start(**arguments)
    service._worker.join(timeout=3)
    assert not service._worker.is_alive()
    status = service.status()
    assert status.phase == CollectionPhase.FINISHED, status.error
    assert status.backend == "fake"
    assert status.total_episodes == status.saved_episodes == 2
    assert status.current_episode == 2
    assert status.episode_phase == "completed"
    assert status.session_elapsed_s > 0
    assert read_ledger(arguments["ledger_path"])[-1]["ok"] is True
    assert service._hardware_lock.holder() is None


def test_service_stop_preserves_saved_episode_and_progress(tmp_path, monkeypatch):
    second_episode = Event()

    class ObservedRecording(FakeRecordingProcess):
        PREPARATION_SECONDS = 0.01

        def _stage(self, phase, duration=0.0):
            if phase == "recording" and self.progress()["saved_episodes"] == 1:
                second_episode.set()
                duration = 30
            return super()._stage(phase, duration)

    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    monkeypatch.setattr(collection_module, "FakeRecordingProcess", ObservedRecording)
    arguments = prepare_start_arguments(request(), data_base=tmp_path)
    service = CollectionService(data_base=tmp_path)
    service.start(**arguments)
    try:
        assert second_episode.wait(3)
    finally:
        service.stop()
        service._worker.join(timeout=3)
    assert not service._worker.is_alive()
    status = service.status()
    assert status.stop_requested is True
    assert status.saved_episodes == 1
    assert status.episode_phase == "stopped"
    record = read_ledger(arguments["ledger_path"])[-1]
    assert record["episodes_after"] == 1
    assert record["interrupted"] is True
    assert service._hardware_lock.holder() is None


def test_fake_command_api_advances_two_episodes_without_hardware(tmp_path, monkeypatch):
    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    monkeypatch.setattr(FakeRecordingProcess, "PREPARATION_SECONDS", 30)

    def hardware_forbidden(*args, **kwargs):
        raise AssertionError("Fake collection must not open real recording hardware")

    monkeypatch.setattr(collection_module, "build_hardware_args", hardware_forbidden)
    monkeypatch.setattr(collection_module, "RecordingProcess", hardware_forbidden)

    app = create_app(data_base=tmp_path)
    body = request()
    body["sessions"][0].update(episode_time_s=30, reset_time_s=30)

    with TestClient(app) as client:
        response = client.post("/api/collection/start", json=body)
        assert response.status_code == 202, response.text

        def wait_for(phase, episode):
            deadline = monotonic() + 3
            while monotonic() < deadline:
                status = client.get("/api/collection/status").json()
                if (
                    status["episode_phase"] == phase
                    and status["current_episode"] == episode
                ):
                    return status
                assert status["phase"] != "failed", status
                sleep(0.01)
            raise AssertionError(f"Never reached {phase}, episode {episode}: {status}")

        def command(status, action, **overrides):
            return client.post("/api/collection/command", json={
                "action": action,
                "recording_id": status["recording_id"],
                "expected_phase": status["episode_phase"],
                "expected_episode": status["current_episode"],
                **overrides,
            })

        preparing = wait_for("preparing", 1)
        assert preparing["recording_id"]
        assert command(preparing, "start_episode", recording_id="old").status_code == 409
        assert command(preparing, "finish_episode").status_code == 409
        response = command(preparing, "start_episode")
        assert response.status_code == 202, response.text
        assert response.json()["status"] == "queued"

        first = wait_for("recording", 1)
        assert command(first, "finish_episode", expected_episode=2).status_code == 409
        assert command(first, "finish_episode").status_code == 202

        resetting = wait_for("resetting", 1)
        assert resetting["saved_episodes"] == 0
        assert command(resetting, "start_episode").status_code == 202

        second = wait_for("recording", 2)
        assert second["saved_episodes"] == 1
        assert command(second, "finish_episode").status_code == 202

        completed = wait_for("completed", 2)
        app.state.collection._worker.join(timeout=3)
        assert not app.state.collection._worker.is_alive()
        final = client.get("/api/collection/status").json()
        assert final["phase"] == "finished", final
        assert completed["saved_episodes"] == final["saved_episodes"] == 2
        assert command(second, "finish_episode").status_code == 409

    root = tmp_path / "fake/collection/raw/progress_demo"
    assert json.loads((root / "meta/info.json").read_text())["total_episodes"] == 2
    assert len((root / "meta/episodes.jsonl").read_text().splitlines()) == 2
    assert not list(root.rglob("*.mp4"))
