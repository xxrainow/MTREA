import json
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest
from fastapi.testclient import TestClient

from lab.core import paths, serving
from lab.core.hardware.lock import HardwareLock
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.server.main import create_app
from robot.config import POLICY_SERVER_PORT


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def client(tmp_path):
    app = create_app(data_base=tmp_path / "data", remote_config=tmp_path / "missing.yaml")
    with TestClient(app) as client:
        executor = Mock()
        executor.start.return_value = 12345
        executor.is_running.return_value = True
        executor.read_log.return_value = "rollout output\n"
        app.state.runner.executors[Executor.LOCAL] = executor
        app.state.cameras.stop = Mock()
        root = tmp_path / "server"
        write(root / "status.json", {"status": "ready", "port": 19234})
        write(root / "session.json", {"checkpoint": "exact checkpoint string"})
        server = Job(id="server", spec=JobSpec(
            kind=JobKind.SERVE_POLICY, cmd="never executed", run_dir=str(root),
            metadata={"checkpoint": "exact checkpoint string", "server_port": 19234},
        ), pid=45678, status=JobStatus.RUNNING)
        app.state.runner.store.save(server)
        yield client


def body(**changes):
    return {"server_job_id": "server", "instruction": 'Hold "block"; $(do-not-run)',
            "max_relative_target": 5.0, "time_limit_s": 30.0, **changes}


def start(client):
    response = client.post("/api/rollouts/start", json=body())
    assert response.status_code == 201, response.text
    value = response.json()
    job = client.app.state.runner.store.get(value["job"]["id"])
    return job, Path(job.spec.run_dir)


def owner(root, **changes):
    write(root / "status.json", {
        "state": "running", "process_alive": True, "pid": os.getpid(),
        "process_started": psutil.Process().create_time(), "trial_token": "test-token",
        "workers_stopped": False, "arm_released": False, **changes,
    })


def test_check_is_read_only_and_start_builds_local_worker(client):
    response = client.post("/api/rollouts/check", json=body())
    assert response.status_code == 200
    assert response.json()["checkpoint"] == "exact checkpoint string"
    assert response.json()["server_address"] == "127.0.0.1:19234"
    executor = client.app.state.runner.executors[Executor.LOCAL]
    executor.start.assert_not_called()
    client.app.state.cameras.stop.assert_not_called()
    job, root = start(client)
    argv = shlex.split(job.spec.cmd)
    assert argv[1:3] == ["-m", "robot.rollout_worker"]
    assert "--checkpoint=exact checkpoint string" in argv
    assert f"--task={body()['instruction']}" in argv
    assert f"--data-dir={client.app.state.data_base}" in argv
    assert f"--trial-dir={root}" in argv
    assert job.spec.executor == Executor.LOCAL
    assert job.spec.metadata["server_job_id"] == "server"
    assert root.parent == client.app.state.rollout.root
    client.app.state.cameras.stop.assert_called_once()
    assert client.post(f"/api/rollouts/{job.id}/stop").status_code == 409
    assert client.post(f"/api/jobs/{job.id}/stop").status_code == 409
    executor.stop.assert_not_called()


def test_commands_acknowledgements_and_generic_jobs_stop_never_kill(client):
    job, root = start(client)
    owner(root)
    assert client.post(f"/api/rollouts/{job.id}/release").status_code == 409
    response = client.post(f"/api/rollouts/{job.id}/stop")
    assert response.status_code == 202
    command_id = response.json()["command_id"]
    command = json.loads((root / "commands" / f"{command_id}.json").read_text())
    assert command["action"] == "stop" and command["trial_token"] == "test-token"
    route = f"/api/rollouts/{job.id}/commands/{command_id}"
    assert client.get(route).json()["state"] == "queued"
    write(root / "command_results" / f"{command_id}.json", {"id": command_id, "state": "completed"})
    assert client.get(route).json()["state"] == "completed"
    assert client.post(f"/api/jobs/{job.id}/stop").status_code == 200
    executor = client.app.state.runner.executors[Executor.LOCAL]
    executor.stop.assert_not_called()
    assert client.app.state.runner.store.get(job.id).status == JobStatus.RUNNING
    owner(root, state="stopped", workers_stopped=True)
    assert client.post(f"/api/rollouts/{job.id}/release").status_code == 202
    assert client.get(f"/api/rollouts/{job.id}/logs").json()["content"] == "rollout output\n"


def test_failed_cleanup_allows_release_retry_with_worker_as_authority(client):
    job, root = start(client)
    owner(root, state="failed", workers_stopped=False)
    # A worker may finish after its join timeout; the live owner checks again.
    assert client.post(f"/api/rollouts/{job.id}/release").status_code == 202


def test_unknown_job_with_live_owner_still_accepts_jobs_stop(client):
    job, root = start(client)
    owner(root)
    job.status = JobStatus.UNKNOWN
    client.app.state.runner.store.save(job)
    assert client.post("/api/rollouts/start", json=body()).status_code == 409
    assert client.post(f"/api/jobs/{job.id}/stop").status_code == 200
    assert len(list((root / "commands").glob("*.json"))) == 1
    client.app.state.runner.executors[Executor.LOCAL].stop.assert_not_called()


def test_stopped_owner_blocks_new_rollout_collection_and_camera_preview(client):
    job, root = start(client)
    owner(root, state="stopped", workers_stopped=True)
    assert client.post("/api/rollouts/start", json=body()).status_code == 409
    assert client.post("/api/rollouts/check", json=body()).status_code == 409
    assert client.post("/api/cameras/start").status_code == 409
    # Use an otherwise valid collection request from the API's input schema.
    from lab.server.collection_api import start_collection
    from fastapi import HTTPException
    request = Mock()
    request.app = client.app
    with pytest.raises(HTTPException) as raised:
        start_collection(Mock(), request)
    assert raised.value.status_code == 409
    assert client.get(f"/api/rollouts/{job.id}").json()["owner_alive"]


def test_sessions_survive_service_recreation_and_dead_owner_is_unknown(client):
    from lab.core.rollout import RolloutService
    job, root = start(client)
    owner(root)
    write(root / "session.json", {"checkpoint": "exact checkpoint string"})
    client.app.state.rollout = RolloutService(client.app.state.data_base)
    sessions = client.get("/api/rollouts").json()
    assert len(sessions) == 1 and sessions[0]["job"]["id"] == job.id
    assert sessions[0]["session"]["checkpoint"] == "exact checkpoint string"
    owner(root, process_started=0)
    detail = client.get(f"/api/rollouts/{job.id}").json()
    assert detail["state"] == "ended_unknown" and not detail["owner_alive"]
    assert client.post(f"/api/rollouts/{job.id}/stop").status_code == 409


@pytest.mark.parametrize("changes", [
    {"instruction": "   "}, {"server_job_id": "missing"},
    {"max_relative_target": 0}, {"max_relative_target": True},
    {"time_limit_s": -1}, {"time_limit_s": "30"}, {"checkpoint": "injected"},
])
def test_invalid_start_never_submits_or_stops_preview(client, changes):
    assert client.post("/api/rollouts/start", json=body(**changes)).status_code in {400, 422}
    client.app.state.runner.executors[Executor.LOCAL].start.assert_not_called()
    client.app.state.cameras.stop.assert_not_called()


def test_not_ready_server_and_busy_hardware_are_refused(client):
    server = client.app.state.runner.store.get("server")
    write(Path(server.spec.run_dir) / "status.json", {"status": "loading"})
    assert client.post("/api/rollouts/start", json=body()).status_code == 400
    lock = HardwareLock(paths.hardware_lock(client.app.state.data_base))
    lock.acquire("teleoperation")
    try:
        assert client.post("/api/rollouts/check", json=body()).status_code == 409
    finally:
        lock.release()
    client.app.state.runner.executors[Executor.LOCAL].start.assert_not_called()


def test_active_collection_and_fake_hardware_refuse_start(client):
    client.app.state.collection.is_active = Mock(return_value=True)
    assert client.post("/api/rollouts/start", json=body()).status_code == 409
    client.app.state.collection.is_active.return_value = False
    client.app.state.hardware.backend = Mock(name="fake-backend")
    client.app.state.hardware.backend.name = "fake"
    assert client.post("/api/rollouts/start", json=body()).status_code == 400
    client.app.state.cameras.stop.assert_not_called()


def test_ssh_requires_existing_tunnel_and_uses_robot_side_port(client, monkeypatch):
    runner = client.app.state.runner
    job = runner.store.get("server")
    from dataclasses import replace
    job.spec = replace(job.spec, executor=Executor.SSH)
    runner.store.save(job)
    session = {"unreachable": None, "read_error": None, "state": "ready",
               "tunnel": None, "status": {"port": 8000}}
    monkeypatch.setattr(serving, "session_status", lambda *_: session)
    assert client.post("/api/rollouts/check", json=body()).status_code == 400
    session["tunnel"] = {"status": "running"}
    result = client.post("/api/rollouts/check", json=body())
    assert result.status_code == 200
    assert result.json()["server_address"] == f"127.0.0.1:{POLICY_SERVER_PORT}"


def test_simultaneous_starts_submit_once(client):
    with ThreadPoolExecutor(max_workers=2) as executor:
        codes = list(executor.map(lambda _: client.post("/api/rollouts/start", json=body()).status_code,
                                  range(2)))
    assert sorted(codes) == [201, 409]
    client.app.state.runner.executors[Executor.LOCAL].start.assert_called_once()


def test_missing_wrong_kind_corrupt_status_and_outside_storage(client, tmp_path):
    assert client.get("/api/rollouts/missing").status_code == 404
    assert client.get("/api/rollouts/server").status_code == 404
    job, root = start(client)
    root.mkdir(parents=True)
    (root / "status.json").write_text("broken")
    detail = client.get(f"/api/rollouts/{job.id}").json()
    assert detail["read_error"]
    assert client.post(f"/api/rollouts/{job.id}/stop").status_code == 503
    assert client.get(f"/api/rollouts/{job.id}/commands/not-a-command").status_code == 404
    assert client.get(f"/api/rollouts/{job.id}/commands/{'a' * 32}").status_code == 404
    from dataclasses import replace
    job.spec = replace(job.spec, run_dir=str(tmp_path / "outside"))
    client.app.state.runner.store.save(job)
    assert client.get(f"/api/rollouts/{job.id}").status_code == 400
