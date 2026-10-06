from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request, HTTPException, Query

from lab.core import paths
from lab.core.executors.local import LocalExecutor
from lab.core.models import Executor
from lab.core.runner import Runner
from lab.core.store import JobStore


def create_app(data_base: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repo_root = Path(__file__).resolve().parents[2]
        resolved_data = paths.ensure_dirs(base=data_base)

        store = JobStore(paths.jobs_ledger(resolved_data))
        local_executor = LocalExecutor(repo_root=repo_root)

        app.state.runner = Runner(
            store=store,
            executors={Executor.LOCAL: local_executor},
            data_base=resolved_data,
            repo_root=repo_root,
        )
        app.state.runner_lock = Lock()

        yield

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

        stopped_job = runner.stop(job_id)

        return stopped_job.to_dict()

app = create_app()
