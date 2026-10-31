from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request, HTTPException, Query

from lab.core import paths
from lab.core.executors.local import LocalExecutor
from lab.core.executors.ssh import SshExecutor
from lab.core.hardware.service import HardwareService
from lab.core.models import Executor
from lab.core.runner import Runner
from lab.core.store import JobStore
from lab.server import hardware_api
from lab.core.hardware.cameras import CameraService
from lab.server import camera_api
from lab.core.collection import CollectionService
from lab.server import collection_api
from lab.server import datasets_api
from lab.server import remote_api
from lab.server import train_api
from lab.core.remote import RemoteError, RemoteNotConfigured, load_remote


def create_app(
    data_base: Path | None = None,
    remote_config: Path | None = None,
    remote_transport: str = "ssh",
) -> FastAPI:
    """remote_config/remote_transport: robot/configs/remote.yaml over SSH unless a test overrides them."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repo_root = Path(__file__).resolve().parents[2]
        resolved_data = paths.ensure_dirs(base=data_base)

        store = JobStore(paths.jobs_ledger(resolved_data))
        local_executor = LocalExecutor(repo_root=repo_root)
        executors = {Executor.LOCAL: local_executor}
        try:
            executors[Executor.SSH] = SshExecutor(load_remote(remote_config, remote_transport))
        except RemoteNotConfigured:
            pass  # ssh jobs show "executor unavailable" until remote.yaml exists

        app.state.runner = Runner(
            store=store,
            executors=executors,
            data_base=resolved_data,
            repo_root=repo_root,
        )
        app.state.runner_lock = Lock()
        
        app.state.hardware = HardwareService(data_base=resolved_data)
        app.state.cameras = CameraService()
        app.state.collection = CollectionService(
            data_base=resolved_data
        )
        app.state.collection_gate = Lock()
        app.state.data_base = resolved_data
        app.state.remote_config = remote_config
        app.state.remote_transport = remote_transport
        app.state.remote_cache = remote_api.TtlCache()
        app.state.repo_root = repo_root
        app.state.train_runs_cache = train_api.runs_cache()

        try:
            yield
        finally:
            try:
                app.state.collection.shutdown()
            finally:
                try:
                    app.state.cameras.stop()
                finally:
                    app.state.hardware.shutdown()

    application = FastAPI(
        title="MTREA Lab",
        lifespan=lifespan,
    )

    application.add_api_route(
        "/api/health", health, methods=["GET"]
    )
    application.add_api_route(
        "/api/jobs", list_jobs, methods=["GET"]
    )
    application.add_api_route(
        "/api/jobs/{job_id}", get_job, methods=["GET"]
    )
    application.add_api_route(
        "/api/jobs/{job_id}/logs", get_job_logs, methods=["GET"]
    )
    application.add_api_route(
        "/api/jobs/{job_id}/stop", stop_job, methods=["POST"]
    )
    application.include_router(hardware_api.router)
    application.include_router(camera_api.router)
    application.include_router(collection_api.router)
    application.include_router(datasets_api.router)
    application.include_router(remote_api.router)
    application.include_router(train_api.router)

    return application

def health() -> dict[str, str]:
    return {"status": "ok"}

def list_jobs(request: Request) -> list[dict]:
    runner: Runner = request.app.state.runner

    with request.app.state.runner_lock:
        jobs = runner.refresh()
        return [job.to_dict() for job in jobs]

def get_job(job_id: str, request: Request) -> dict:
    runner: Runner = request.app.state.runner

    with request.app.state.runner_lock:
        job = runner.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            )

        return job.to_dict()

def get_job_logs(
    job_id: str,
    request: Request,
    tail_lines: int = Query(default=100, ge=1, le=1000),
) -> dict:
    runner: Runner = request.app.state.runner

    with request.app.state.runner_lock:
        job = runner.store.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            )

        content = runner.read_log(
            job_id,
            tail_lines=tail_lines,
        )

        return {
            "job_id": job_id,
            "content": content,
        }

def stop_job(job_id: str, request: Request) -> dict:
    runner: Runner = request.app.state.runner

    with request.app.state.runner_lock:
        job = runner.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            )

        if job.status.value == "running":
            if job.spec.executor not in runner.executors:
                raise HTTPException(
                    status_code=409,
                    detail="Executor unavailable for this job",
                )

        try:
            stopped_job = runner.stop(job_id)
        except RemoteError as e:
            raise HTTPException(
                status_code=503,
                detail=f"GPU host unreachable; the job may still be running: {e}",
            ) from None

        return stopped_job.to_dict()

app = create_app()
