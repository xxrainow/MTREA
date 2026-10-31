"""Training from the Train page. Delegates to lab.core.training; stop stays POST /api/jobs/{id}/stop."""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from lab.core import training
from lab.core.executors.ssh import SshExecutor
from lab.core.models import Executor, Job
from lab.core.remote import RemoteError
from lab.core.runner import JobConflict
from lab.server.remote_api import TtlCache, get_remote, remote_status

router = APIRouter(prefix="/api/train", tags=["train"])

RUNS_CACHE_S = 3.0


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["real", "fake"]
    name: str


class TrainBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["ssh", "local"]
    fake: bool = False
    dataset: DatasetRef
    run_name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def to_request(self) -> training.TrainRequest:
        return training.TrainRequest(target=self.target, dataset_backend=self.dataset.backend,
                                     dataset_name=self.dataset.name, run_name=self.run_name,
                                     fake=self.fake, params=self.params)


def _spec(request: Request) -> dict:
    try:
        return training.describe(request.app.state.repo_root)
    except Exception as e:  # noqa: BLE001 - surfaced as-is: the form cannot render without it
        raise HTTPException(status_code=500, detail=f"Cannot describe training options: {e}") from None


def _context(request: Request) -> dict:
    remote, why = get_remote(request)
    return {"spec": _spec(request), "remote": remote, "data_base": request.app.state.data_base,
            "repo_root": request.app.state.repo_root, "not_configured": why}


def _train_job(request: Request, job_id: str, refresh: bool) -> Job:
    # Only status needs a fresh job state; each refresh of an ssh job is a round trip.
    runner = request.app.state.runner
    with request.app.state.runner_lock:
        job = runner.get(job_id) if refresh else runner.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _run_dir(request: Request, job: Job) -> training.RunDir:
    remote, _ = get_remote(request)
    try:
        return training.run_dir_of(job, remote)
    except LookupError:
        raise HTTPException(status_code=404, detail="Not a training run") from None
    except RemoteError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


def _read(request: Request, job_id: str, read, refresh: bool = False):
    job = _train_job(request, job_id, refresh)
    run_dir = _run_dir(request, job)
    try:
        return read(job, run_dir)
    except RemoteError as e:
        raise HTTPException(status_code=503, detail=f"GPU host unreachable: {e}") from None


@router.get("/describe")
def describe(request: Request) -> dict:
    return _spec(request)


@router.get("/targets")
def targets(request: Request) -> dict:
    ssh = remote_status(request)
    return {"local": True, "ssh": {k: ssh[k] for k in ("configured", "reachable", "host", "message")}}


@router.post("/check")
def check(body: TrainBody, request: Request) -> dict:
    try:
        return training.check(body.to_request(), **_context(request))
    except training.TrainError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


@router.post("/start", status_code=201)
def start(body: TrainBody, request: Request) -> dict:
    context = _context(request)
    state = request.app.state
    remote = context["remote"]
    if body.target == "ssh" and remote is not None and Executor.SSH not in state.runner.executors:
        # remote.yaml was added after the server started.
        state.runner.executors[Executor.SSH] = SshExecutor(remote)
    try:
        job = training.start(body.to_request(), runner=state.runner, lock=state.runner_lock, **context)
    except training.TrainError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except training.CheckFailed as e:
        raise HTTPException(status_code=409, detail={"message": "Checks failed", **e.result}) from None
    except JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    except RemoteError as e:
        raise HTTPException(status_code=503, detail=f"GPU host unreachable: {e}") from None
    state.train_runs_cache.clear()
    return job.to_dict()


@router.get("/runs")
def runs(request: Request) -> list[dict]:
    def compute() -> list[dict]:
        with request.app.state.runner_lock:
            jobs = request.app.state.runner.refresh()
        remote, _ = get_remote(request)
        return training.list_runs(jobs, remote)

    return request.app.state.train_runs_cache.get("runs", compute)


@router.get("/runs/{job_id}/status")
def run_status(job_id: str, request: Request) -> dict:
    return _read(request, job_id, training.run_status, refresh=True)


@router.get("/runs/{job_id}/metrics")
def run_metrics(job_id: str, request: Request, offset: int = Query(default=0, ge=0)) -> dict:
    return _read(request, job_id, lambda job, run_dir: training.read_metrics(run_dir, offset))


@router.get("/runs/{job_id}/checkpoints")
def run_checkpoints(job_id: str, request: Request) -> dict:
    return _read(request, job_id, lambda job, run_dir: {
        "run_dir": job.spec.run_dir, "host": job.spec.host,
        "checkpoints": training.read_json(run_dir, "checkpoints.json")})


@router.get("/runs/{job_id}/config")
def run_config(job_id: str, request: Request) -> dict:
    return _read(request, job_id, lambda job, run_dir: {"config": training.read_json(run_dir, "config.json")})


@router.get("/runs/{job_id}/result")
def run_result(job_id: str, request: Request) -> dict:
    return _read(request, job_id, lambda job, run_dir: {"result": training.read_json(run_dir, "result.json")})


def runs_cache() -> TtlCache:
    return TtlCache(RUNS_CACHE_S)
