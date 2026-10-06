from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request, HTTPException, Query

from lab.core import paths
from lab.core.executors.local import LocalExecutor
from lab.core.models import Executor
from lab.core.runner import Runner
from lab.core.store import JobStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo_root = Path(__file__).resolve().parents[2]
    data_base = paths.ensure_dirs()

    store = JobStore(paths.jobs_ledger(data_base))
    local_executor = LocalExecutor(repo_root=repo_root)

    runner = Runner(
        store=store,
        executors={Executor.LOCAL: local_executor},
        data_base=data_base,
        repo_root=repo_root,
    )

    app.state.runner = runner
    app.state.runner_lock = Lock()

    yield


app = FastAPI(
    title="MTREA Lab",
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs(request: Request) -> list[dict]:
    runner: Runner = request.app.state.runner

    with request.app.state.runner_lock:
        jobs = runner.refresh()
        return [job.to_dict() for job in jobs]

@app.get("/api/jobs/{job_id}")
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


@app.get("/api/jobs/{job_id}/logs")
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

@app.post("/api/jobs/{job_id}/stop")
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
