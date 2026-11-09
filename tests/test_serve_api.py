"""Serving API checks without remote connections or model execution."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from lab.core.executors.ssh import SshExecutor
from lab.core.models import Executor, Job, JobKind, JobSpec
from lab.core.remote import Remote, RemoteConfig
from lab.server.main import create_app
from lab.server import serve_api


@pytest.fixture
def client(tmp_path):
    app = create_app(
        data_base=tmp_path / "data",
        remote_config=tmp_path / "no-remote.yaml",
    )

    with TestClient(app) as test_client:
        yield test_client


def test_sessions_are_empty_without_server_jobs(client):
    response = client.get("/api/serve/sessions")

    assert response.status_code == 200
    assert response.json() == []


def test_real_serving_on_local_machine_is_refused(client):
    response = client.post(
        "/api/serve/start",
        json={
            "target": "local",
            "mode": "real",
        },
    )

    assert response.status_code == 400
    assert "fake hold only" in response.json()["detail"]


def test_tunnel_for_missing_session_is_not_found(client):
    response = client.post(
        "/api/serve/sessions/no-such-job/tunnel"
    )

    assert response.status_code == 404


@pytest.fixture
def configured_remote():
    return Remote(RemoteConfig(
        host="gpu.example", user="lab", repo_dir="/srv/mtrea",
        data_dir="/srv/mtrea/data", python="/srv/venv/bin/python",
        policy_port=8000,
    ))


@pytest.mark.parametrize("change", [
    {"host": "other-gpu.example"},
    {"repo_dir": "/srv/other-checkout"},
    {"python": "/srv/other-venv/bin/python"},
    {"policy_port": 8001},
])
def test_changed_remote_config_is_refused_before_execution(
    client, monkeypatch, configured_remote, change,
):
    executor = SshExecutor(configured_remote)
    client.app.state.runner.executors[Executor.SSH] = executor
    changed = Remote(replace(configured_remote.config, **change))
    monkeypatch.setattr(serve_api, "get_remote", lambda request: (changed, ""))
    start = Mock(side_effect=AssertionError("Serving must not start with changed settings"))
    monkeypatch.setattr(serve_api.serving, "start", start)

    response = client.post("/api/serve/start", json={"target": "ssh", "mode": "real"})

    assert response.status_code == 400
    assert "configuration changed" in response.json()["detail"]
    assert "restart the lab backend" in response.json()["detail"]
    start.assert_not_called()
    assert client.app.state.runner.executors[Executor.SSH] is executor


def test_equal_remote_config_reuses_executor(client, monkeypatch, configured_remote):
    executor = SshExecutor(configured_remote)
    client.app.state.runner.executors[Executor.SSH] = executor
    reloaded = Remote(replace(configured_remote.config))
    monkeypatch.setattr(serve_api, "get_remote", lambda request: (reloaded, ""))
    job = Job(id="serve-test", spec=JobSpec(kind=JobKind.SERVE_POLICY, cmd="not executed"))
    start = Mock(return_value=job)
    monkeypatch.setattr(serve_api.serving, "start", start)

    response = client.post("/api/serve/start", json={"target": "ssh", "mode": "real"})

    assert response.status_code == 201
    assert response.json()["id"] == job.id
    start.assert_called_once()
    assert start.call_args.kwargs["remote"] is reloaded
    assert client.app.state.runner.executors[Executor.SSH] is executor
