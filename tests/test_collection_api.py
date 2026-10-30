#--- Replace the collection service and camera with a FAKE

import json

import pytest
from fastapi.testclient import TestClient

import lab.core.collection_input as collection_input
from lab.core.models import CollectionPhase, CollectionStatus
from lab.server.main import create_app


class FakeCollectionService:
    def __init__(self):
        self.calls = []
        self.current = CollectionStatus()

    def status(self):
        return self.current

    def is_active(self):
        return self.current.phase in {
            CollectionPhase.STARTING,
            CollectionPhase.STOPPING,
        }

    def start(self, **arguments):
        self.calls.append(arguments)
        self.current = CollectionStatus(
            phase=CollectionPhase.STARTING,
            session_index=arguments["session_index"],
        )
        return self.current

    def stop(self):
        self.current = CollectionStatus(
            phase=CollectionPhase.STOPPING,
            session_index=self.current.session_index,
            stop_requested=True,
        )
        return self.current

    def shutdown(self):
        pass


class FakeCameras:
    def __init__(self):
        self.running = True

    def start(self):
        self.running = True
        return {"running": True, "cameras": [], "error": None}

    def stop(self):
        self.running = False
        return {"running": False, "cameras": [], "error": None}


@pytest.fixture
def api(tmp_path, monkeypatch):
    original_version = collection_input.metadata.version

    monkeypatch.setattr(
        collection_input.metadata,
        "version",
        lambda name: (
            "fake"
            if name == "lerobot"
            else original_version(name)
        ),
    )

    app = create_app(data_base=tmp_path)

    with TestClient(app) as client:
        service = FakeCollectionService()
        cameras = FakeCameras()
        app.state.collection = service
        app.state.cameras = cameras
        yield client, service, cameras, tmp_path


def payload():
    return {
        "dataset": "api_demo",
        "session_index": 1,
        "sessions": [
            {
                "instruction": "Pick up the red block.",
                "num_episodes": 2,
                "episode_time_s": 20,
                "reset_time_s": 5,
            },
        ],
    }


def test_start_stops_preview_and_passes_saved_input(api):
    client, service, cameras, base = api

    response = client.post(
        "/api/collection/start",
        json=payload(),
    )

    assert response.status_code == 202, response.text
    assert response.json()["phase"] == "starting"
    assert cameras.running is False
    assert len(service.calls) == 1

    arguments = service.calls[0]
    assert arguments["instruction"] == "Pick up the red block."
    assert arguments["planned_episodes"] == 2
    assert arguments["session_index"] == 1
    assert arguments["dataset_root"] == base / "raw" / "api_demo"

    config_path = base / "raw" / "api_demo.collection.input.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "seed" not in saved
    assert "seed" not in arguments["record_context"]
    assert saved["sessions"][0]["num_episodes"] == 2

    assert not arguments["dataset_root"].exists()


def test_active_collection_rejects_start_and_preview(api):
    client, service, cameras, _ = api

    first = client.post("/api/collection/start", json=payload())
    assert first.status_code == 202

    duplicate = client.post("/api/collection/start", json=payload())
    preview = client.post("/api/cameras/start")

    assert duplicate.status_code == 409
    assert preview.status_code == 409
    assert len(service.calls) == 1
    assert cameras.running is False


def test_stop_returns_stop_requested_state(api):
    client, _, _, _ = api

    started = client.post("/api/collection/start", json=payload())
    assert started.status_code == 202

    stopped = client.post("/api/collection/stop")
    status = client.get("/api/collection/status")

    assert stopped.status_code == 200
    assert stopped.json()["phase"] == "stopping"
    assert status.json()["stop_requested"] is True


def test_invalid_input_does_not_start_collection(api):
    client, service, cameras, base = api
    body = payload()
    body["sessions"][0]["num_episodes"] = 0

    response = client.post("/api/collection/start", json=body)

    assert response.status_code == 422
    assert service.calls == []
    assert cameras.running is True
    assert not (
        base / "raw" / "api_demo.collection.input.json"
    ).exists()


def test_seed_is_no_longer_a_collection_input(api):
    client, service, cameras, _ = api
    body = payload()
    body["seed"] = 42
    response = client.post("/api/collection/start", json=body)
    assert response.status_code == 422
    assert service.calls == []
    assert cameras.running is True


def test_legacy_seed_snapshot_keeps_hash_and_rejects_other_changes(tmp_path):
    import hashlib

    from lab.core.collection_input import save_collection_input

    body = payload()
    initial = save_collection_input(body, data_base=tmp_path)
    path = initial["config_path"]
    snapshot = json.loads(path.read_text())
    snapshot["seed"] = 42
    legacy = json.dumps(snapshot, indent=2) + "\n"
    path.write_text(legacy)

    resumed = save_collection_input(body, data_base=tmp_path)
    assert resumed["plan_sha"] == hashlib.sha256(legacy.encode()).hexdigest()
    assert path.read_text() == legacy
    assert "seed" not in resumed

    body["sessions"][0]["instruction"] = "Different task."
    with pytest.raises(ValueError, match="Collection settings changed"):
        save_collection_input(body, data_base=tmp_path)
