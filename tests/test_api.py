from fastapi.testclient import TestClient

from lab.server.main import create_app
from lab.core import paths
from lab.core.models import Executor, Job, JobKind, JobSpec, JobStatus
from lab.core.store import JobStore

import shlex
import sys
import time


def test_health(tmp_path):
    app = create_app(data_base=tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_jobs(tmp_path):
    app = create_app(data_base=tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/jobs")

    assert response.status_code == 200
    assert response.json() == []


def test_missing_job(tmp_path):
    app = create_app(data_base=tmp_path)

    with TestClient(app) as client:
        detail = client.get("/api/jobs/missing")
        logs = client.get("/api/jobs/missing/logs")
        stop = client.post("/api/jobs/missing/stop")

    assert detail.status_code == 404
    assert logs.status_code == 404
    assert stop.status_code == 404


def test_invalid_log_limit(tmp_path):
    app = create_app(data_base=tmp_path)

    with TestClient(app) as client:
        response = client.get(
            "/api/jobs/missing/logs",
            params={"tail_lines": 0},
        )

    assert response.status_code == 422


def test_job_detail_and_logs(tmp_path):
    data_base = paths.ensure_dirs(base=tmp_path)

    job_id = "test-job-001"
    log_path = paths.job_log(job_id, base=data_base)
    log_path.write_text(
        "first line\nsecond line\nthird line\n",
        encoding="utf-8",
    )

    job = Job(
        id=job_id,
        spec=JobSpec(
            kind=JobKind.TRAIN,
            cmd="test-only",
            label="API log test",
        ),
        status=JobStatus.DONE,
        exit_code=0,
        log_path=str(log_path),
    )

    store = JobStore(paths.jobs_ledger(base=data_base))
    store.save(job)

    app = create_app(data_base=data_base)

    with TestClient(app) as client:
        detail_response = client.get(f"/api/jobs/{job_id}")

        logs_response = client.get(
            f"/api/jobs/{job_id}/logs",
            params={"tail_lines": 2},
        )

    assert detail_response.status_code == 200

    detail = detail_response.json()
    assert detail["id"] == job_id
    assert detail["status"] == "done"
    assert detail["exit_code"] == 0

    assert logs_response.status_code == 200
    assert logs_response.json() == {
        "job_id": job_id,
        "content": "second line\nthird line",
    }

def test_stop_running_job(tmp_path):
    app = create_app(data_base=tmp_path)

    with TestClient(app) as client:
        runner = app.state.runner
        executor = runner.executors[Executor.LOCAL]

        code = (
            "import time; "
            "print('ready', flush=True); "
            "time.sleep(60)"
        )
        command = "exec " + shlex.join([
            sys.executable,
            "-u",
            "-c",
            code,
        ])

        job = runner.submit(
            JobSpec(
                kind=JobKind.TRAIN,
                cmd=command,
                label="Stop API test",
            )
        )

        try:
            deadline = time.monotonic() + 5

            while "ready" not in runner.read_log(job.id):
                if time.monotonic() >= deadline:
                    raise AssertionError("Test process did not become ready")
                time.sleep(0.05)

            assert executor.is_running(job.pid)

            response = client.post(
                f"/api/jobs/{job.id}/stop"
            )

            assert response.status_code == 200

            body = response.json()
            assert body["id"] == job.id
            assert body["status"] == "stopped"
            assert body["finished_at"] is not None

            assert not executor.is_running(job.pid)

            saved_job = runner.store.get(job.id)
            assert saved_job is not None
            assert saved_job.status == JobStatus.STOPPED

            again = client.post(
                f"/api/jobs/{job.id}/stop"
            )
            assert again.status_code == 200
            assert again.json()["status"] == "stopped"

        finally:
            executor.stop(job.pid)
            
            
def test_jobs_survive_app_restart(tmp_path):
    job_id = "restart-test-job"

    first_app = create_app(data_base=tmp_path)

    with TestClient(first_app):
        runner = first_app.state.runner

        log_path = paths.job_log(job_id, base=tmp_path)
        log_path.write_text(
            "saved before restart\n",
            encoding="utf-8",
        )

        job = Job(
            id=job_id,
            spec=JobSpec(
                kind=JobKind.TRAIN,
                cmd="test-only",
                label="Restart test",
            ),
            status=JobStatus.DONE,
            exit_code=0,
            log_path=str(log_path),
        )

        runner.store.save(job)

    # Start a new app with the same data directory.
    second_app = create_app(data_base=tmp_path)

    with TestClient(second_app) as client:
        assert second_app.state.runner is not runner

        list_response = client.get("/api/jobs")
        detail_response = client.get(f"/api/jobs/{job_id}")
        logs_response = client.get(f"/api/jobs/{job_id}/logs")

    assert list_response.status_code == 200
    assert [
        item["id"] for item in list_response.json()
    ] == [job_id]

    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "done"
    assert detail_response.json()["exit_code"] == 0

    assert logs_response.status_code == 200
    assert logs_response.json()["content"] == "saved before restart"
