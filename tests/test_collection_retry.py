import json
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

from lab.core.fake_recorder import FakeRecordingProcess
from lab.server.main import create_app


def wait_for(client, phase, episode, *, previous_phase_id=None):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        status = client.get("/api/collection/status").json()
        if (
            status["episode_phase"] == phase
            and status["current_episode"] == episode
            and status["phase_id"] != previous_phase_id
        ):
            return status
        assert status["phase"] != "failed", status
        sleep(0.005)
    raise AssertionError(f"Did not reach {phase}, episode {episode}: {status}")


def command(client, status, action):
    return client.post("/api/collection/command", json={
        "action": action,
        "recording_id": status["recording_id"],
        "expected_phase_id": status["phase_id"],
        "expected_phase": status["episode_phase"],
        "expected_episode": status["current_episode"],
    })


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    monkeypatch.setattr(FakeRecordingProcess, "PREPARATION_SECONDS", 30)
    app = create_app(data_base=tmp_path)
    with TestClient(app) as client:
        yield client, app, tmp_path / "fake/collection/raw/retry_demo"


def start(client, reset):
    response = client.post("/api/collection/start", json={
        "dataset": "retry_demo",
        "session_index": 1,
        "sessions": [{
            "instruction": "Move the block.",
            "num_episodes": 2,
            "episode_time_s": 30,
            "reset_time_s": reset,
        }],
    })
    assert response.status_code == 202, response.text
    preparing = wait_for(client, "preparing", 1)
    assert command(client, preparing, "start_episode").status_code == 202


@pytest.mark.parametrize("reset", [0, 30])
def test_retry_first_and_last_episode_preserves_saved_data(collection, reset):
    client, app, root = collection
    start(client, reset)

    for episode in (1, 2):
        current = wait_for(client, "recording", episode)
        assert current["saved_episodes"] == episode - 1
        assert command(client, current, "retry_episode").status_code == 202

        if reset:
            resetting = wait_for(client, "resetting", episode)
            assert resetting["saved_episodes"] == episode - 1
            assert command(client, resetting, "retry_episode").status_code == 409
            assert command(client, resetting, "start_episode").status_code == 202

        retried = wait_for(
            client, "recording", episode,
            previous_phase_id=current["phase_id"],
        )
        assert retried["saved_episodes"] == episode - 1
        rows = (root / "meta/episodes.jsonl").read_text().splitlines()
        assert len(rows) == episode - 1
        assert command(client, current, "retry_episode").status_code == 409
        assert command(client, retried, "finish_episode").status_code == 202

        if episode == 1 and reset:
            resetting = wait_for(client, "resetting", episode)
            assert command(client, resetting, "start_episode").status_code == 202

    wait_for(client, "completed", 2)
    app.state.collection._worker.join(timeout=3)
    assert not app.state.collection._worker.is_alive()
    final = client.get("/api/collection/status").json()
    assert final["phase"] == "finished", final
    assert final["saved_episodes"] == 2
    rows = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in rows] == [0, 1]


def test_stop_during_retry_reset_keeps_previous_saved_episode(collection):
    client, app, root = collection
    start(client, 30)
    first = wait_for(client, "recording", 1)
    assert command(client, first, "finish_episode").status_code == 202
    resetting = wait_for(client, "resetting", 1)
    assert command(client, resetting, "start_episode").status_code == 202
    second = wait_for(client, "recording", 2)
    assert command(client, second, "retry_episode").status_code == 202
    wait_for(client, "resetting", 2)
    assert client.post("/api/collection/stop").status_code == 200
    app.state.collection._worker.join(timeout=3)
    assert not app.state.collection._worker.is_alive()
    final = client.get("/api/collection/status").json()
    assert final["stop_requested"]
    assert final["phase"] == "finished", final
    assert final["error"] is None
    assert final["saved_episodes"] == 1
    assert len((root / "meta/episodes.jsonl").read_text().splitlines()) == 1
    assert app.state.collection._hardware_lock.holder() is None
    from lab.core.collection_ledger import read_ledger, prepare_session

    ledger_path = root.parent / "retry_demo.collection.jsonl"
    entry = read_ledger(ledger_path)[-1]
    assert entry["status"] == "interrupted"
    state = prepare_session(
        dataset_root=root,
        ledger_path=ledger_path,
        plan_sha=entry["plan_sha256"],
        session_index=1,
        planned_episodes=2,
    )
    assert state.remaining_episodes == 1
    assert state.resume
