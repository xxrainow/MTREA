"""HTTP endpoints for policy server preparation."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from lab.core import serving
from lab.core.executors.ssh import SshExecutor
from lab.core.models import Executor, JobKind
from lab.core.remote import RemoteError, RemoteNotConfigured
from lab.core.runner import JobConflict
from lab.server.remote_api import get_remote


router = APIRouter(prefix="/api/serve", tags=["serve"])


class ServeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal["ssh", "local"]
    mode: Literal["real", "hold"]
    source_job_id: str | None = None
    dataset_name: str | None = None
    
    
@router.post("/start", status_code=201)
def start(body: ServeBody, request: Request) -> dict:
    remote, _ = get_remote(request)
    state = request.app.state

    try:
        with state.runner_lock:
            if (
                body.target == "ssh"
                and remote is not None
                and Executor.SSH not in state.runner.executors
            ):
                state.runner.executors[Executor.SSH] = SshExecutor(remote)

            job = serving.start(
                **body.model_dump(),
                runner=state.runner,
                remote=remote,
                data_base=state.data_base,
            )

        return job.to_dict()

    except (ValueError, FileNotFoundError, RemoteNotConfigured) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    except JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    except (RemoteError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("/checkpoints")
def checkpoints(request: Request) -> dict:
    remote, _ = get_remote(request)
    state = request.app.state

    with state.runner_lock:
        jobs = state.runner.refresh()

    return serving.checkpoint_options(jobs, remote)

@router.get("/sessions")
def sessions(request: Request) -> list[dict]:
    remote, _ = get_remote(request)
    state = request.app.state

    with state.runner_lock:
        jobs = state.runner.refresh()

    return [
        serving.session_status(job, jobs, remote)
        for job in jobs
        if job.spec.kind == JobKind.SERVE_POLICY
    ]


@router.post("/sessions/{job_id}/tunnel", status_code=201)
def tunnel(job_id: str, request: Request) -> dict:
    remote, _ = get_remote(request)
    state = request.app.state

    try:
        with state.runner_lock:
            job = state.runner.get(job_id)

            if job is None or job.spec.kind != JobKind.SERVE_POLICY:
                raise HTTPException(
                    status_code=404,
                    detail="Policy session not found",
                )

            tunnel_job = serving.start_tunnel(
                job=job,
                runner=state.runner,
                remote=remote,
            )

        return tunnel_job.to_dict()

    except (ValueError, RemoteNotConfigured) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    except JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    except (RemoteError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
