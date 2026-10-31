"""/api/train against real fake runs of research/train.py on this machine. No network, no GPU."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.core import paths
from lab.server.main import create_app

QUANTILES = {f: {"mean": [0.0], "q01": [-1.0], "q99": [1.0]} for f in ("observation.state", "action")}


@pytest.fixture
def base(tmp_path) -> Path:
    b = tmp_path / "data"
    root = paths.raw_dir(b) / "pick_v1"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1, "fps": 30}))
    (root / "meta" / "stats.json").write_text(json.dumps(QUANTILES))
    return b


@pytest.fixture
def client(base, tmp_path):
    # A missing remote.yaml: this machine's real one must not be used by tests.
    with TestClient(create_app(base, remote_config=tmp_path / "no-remote.yaml")) as c:
        yield c


def body(run_name="h0_pick_v1_s0", **params):
    return {"target": "local", "fake": True, "dataset": {"backend": "real", "name": "pick_v1"}, "run_name": run_name,
            "params": {"method": "h0", "seed": 0, "steps": 40, "save_freq": 10, "warmup_steps": 5, "log_freq": 5,
                       "fake_seconds_per_step": 0.01, **params}}


def wait_done(client, job_id, timeout_s=60) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get(f"/api/train/runs/{job_id}/status").json()
        if status["job"]["status"] != "running":
            return status
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_describe_and_targets(client, base, tmp_path):
    describe = client.get("/api/train/describe").json()
    assert describe["schema_version"] == 1 and {m["name"] for m in describe["methods"]} >= {"h0"}
    targets = client.get("/api/train/targets").json()
    assert targets["local"] is True
    assert targets["ssh"]["configured"] is False and "not found" in targets["ssh"]["message"]

    config = tmp_path / "remote.yaml"
    config.write_text(f"host: gpu\nuser: lab\nrepo_dir: /srv/mtrea\ndata_dir: {(tmp_path / 'gpu').as_posix()}\n"
                      f"python: {Path(sys.executable).as_posix()}\n", encoding="utf-8")
    with TestClient(create_app(base, remote_config=config, remote_transport="local")) as configured:
        ssh = configured.get("/api/train/targets").json()["ssh"]
        assert ssh == {"configured": True, "reachable": True, "host": "lab@gpu", "message": "local transport (no SSH)"}


def test_check_reports_target_checks_and_refusals(client):
    ok = client.post("/api/train/check", json=body()).json()
    assert ok["ok"] is True and ok["target"] == "local" and ok["host"] is None
    assert ok["run_dir"].replace("\\", "/").endswith("fake/runs/h0_pick_v1_s0")
    assert "-m research.train" in ok["command"] and "--fake" in ok["command"]
    real = client.post("/api/train/check", json={**body(), "fake": False,
                                                 "params": {"method": "h0", "seed": 0}}).json()
    assert real["ok"] is False and real["checks"][0]["id"] == "target"
    ssh = client.post("/api/train/check", json={**body(), "target": "ssh"}).json()
    assert ssh["ok"] is False and ssh["checks"][0]["id"] == "remote"
    assert client.post("/api/train/check", json=body(nope=1)).status_code == 400
    assert client.post("/api/train/check", json={**body(), "extra": 1}).status_code == 422


def test_start_poll_metrics_checkpoints_result(client):
    started = client.post("/api/train/start", json=body())
    assert started.status_code == 201, started.text
    job = started.json()
    assert job["spec"]["kind"] == "train" and job["spec"]["label"] == "h0_pick_v1_s0"
    assert job["spec"]["run_dir"] and job["spec"]["host"] is None
    status = wait_done(client, job["id"])
    assert status["job"]["status"] == "done" and status["state"] == "done"
    assert status["status"]["step"] == 40 and status["status"]["fake"] is True

    page = client.get(f"/api/train/runs/{job['id']}/metrics", params={"offset": 0}).json()
    assert [m["step"] for m in page["metrics"]] == [5, 10, 15, 20, 25, 30, 35, 40]
    rest = client.get(f"/api/train/runs/{job['id']}/metrics", params={"offset": page["offset"]}).json()
    assert rest == {"metrics": [], "offset": page["offset"]}

    checkpoints = client.get(f"/api/train/runs/{job['id']}/checkpoints").json()
    assert [c["step"] for c in checkpoints["checkpoints"]["checkpoints"]] == [40]  # older ones pruned
    assert client.get(f"/api/train/runs/{job['id']}/result").json()["result"]["status"] == "done"
    assert client.get(f"/api/train/runs/{job['id']}/config").json()["config"]["fake"] is True

    runs = client.get("/api/train/runs").json()
    assert [(r["job"]["id"], r["state"], r["checkpoint_count"]) for r in runs] == [(job["id"], "done", 1)]
    # the same name again: train.sh --check refuses a non-empty run directory
    again = client.post("/api/train/start", json=body())
    assert again.status_code == 409
    assert any(c["id"] == "output" and c["status"] == "fail" for c in again.json()["detail"]["checks"])


def test_second_start_conflicts_and_stop_records_stopped(client):
    slow = client.post("/api/train/start", json=body("slow", steps=600, log_freq=10, fake_seconds_per_step=0.05)).json()
    try:
        conflict = client.post("/api/train/start", json=body("second"))
        assert conflict.status_code == 409 and "train job" in conflict.json()["detail"]
        # Page metrics by offset while the run is going: no repeats, no gaps.
        url, steps, offset = f"/api/train/runs/{slow['id']}/metrics", [], 0
        deadline = time.monotonic() + 30
        while len(steps) < 6:
            assert time.monotonic() < deadline
            page = client.get(url, params={"offset": offset}).json()
            steps += [m["step"] for m in page["metrics"]]
            offset = page["offset"]
            time.sleep(0.3)
        assert steps == list(range(10, 10 * len(steps) + 1, 10))
    finally:
        stopped = client.post(f"/api/jobs/{slow['id']}/stop")
    assert stopped.status_code == 200 and stopped.json()["status"] == "stopped"
    status = client.get(f"/api/train/runs/{slow['id']}/status").json()
    assert status["state"] == "stopped" and status["job"]["status"] == "stopped"
    assert client.get("/api/train/runs/not-a-job/status").status_code == 404


def test_non_training_jobs_and_old_ledgers(base, tmp_path):
    ledger = paths.jobs_ledger(base)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    old = {"id": "20261001-100000-train-aaaaaa", "status": "done", "pid": 1, "exit_code": 0,
           "submitted_at": "2026-10-01T10:00:00", "started_at": None, "finished_at": None, "log_path": None,
           "git_commit": "abc", "spec": {"kind": "train", "cmd": "scripts/train.sh", "executor": "local",
                                         "label": "old", "cwd": None}}
    ledger.write_text(json.dumps(old) + "\n", encoding="utf-8")
    with TestClient(create_app(base, remote_config=tmp_path / "no-remote.yaml")) as client:
        (job,) = client.get("/api/jobs").json()
        assert job["spec"]["run_dir"] is None and job["spec"]["host"] is None
        assert client.get("/api/train/runs").json() == []  # no run directory, nothing to read
        assert client.get(f"/api/train/runs/{old['id']}/status").status_code == 404
