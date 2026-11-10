"""HTTP adapter for local robot-owner jobs and their command channel."""

from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from lab.core.hardware import HardwareBusy
from lab.core.models import RolloutConfig
from lab.core.remote import RemoteError
from lab.core.runner import JobConflict
from lab.server.remote_api import get_remote

router = APIRouter(prefix="/api/rollouts", tags=["rollouts"])


class RolloutBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_job_id: str = Field(strict=True, min_length=1)
    instruction: str = Field(strict=True, min_length=1)
    max_relative_target: float = Field(strict=True, gt=0, allow_inf_nan=False)
    time_limit_s: float = Field(strict=True, gt=0, allow_inf_nan=False)

    def config(self):
        return RolloutConfig(**self.model_dump())


@contextmanager
def errors():
    try:
        yield
    except KeyError:
        raise HTTPException(404, "Rollout or command not found") from None
    except (HardwareBusy, JobConflict) as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except (OSError, RemoteError) as exc:
        raise HTTPException(503, str(exc)) from None


@router.post("/check")
def check(body: RolloutBody, request: Request):
    state = request.app.state
    remote, _ = get_remote(request)
    with errors(), state.runner_lock:
        return state.rollout.check(body.config(), state.runner, remote)


@router.post("/start", status_code=201)
def start(body: RolloutBody, request: Request):
    state = request.app.state
    remote, _ = get_remote(request)
    with errors(), state.collection_gate, state.runner_lock:
        if state.collection.is_active():
            raise HardwareBusy("Stop collection before starting a rollout")
        if state.hardware.backend.name != "real":
            raise ValueError("Robot rollout is unavailable in fake hardware mode")
        job = state.rollout.start(body.config(), state.runner, remote,
                                  prepare_devices=state.cameras.stop)
        return state.rollout.snapshot(job)


@router.get("")
def sessions(request: Request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout.sessions(state.runner)


@router.get("/{job_id}")
def status(job_id: str, request: Request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout.snapshot(state.rollout.get_job(job_id, state.runner))


def command(job_id, action, request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout.command(job_id, action, state.runner)


@router.post("/{job_id}/stop", status_code=202)
def stop(job_id: str, request: Request):
    return command(job_id, "stop", request)


@router.post("/{job_id}/release", status_code=202)
def release(job_id: str, request: Request):
    return command(job_id, "release", request)


@router.get("/{job_id}/commands/{command_id}")
def result(job_id: str, command_id: str, request: Request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout.command_result(job_id, command_id, state.runner)


@router.get("/{job_id}/logs")
def logs(job_id: str, request: Request,
         tail_lines: int = Query(default=100, ge=1, le=1000)):
    state = request.app.state
    with errors(), state.runner_lock:
        state.rollout.get_job(job_id, state.runner)
        return {"job_id": job_id,
                "content": state.runner.read_log(job_id, tail_lines=tail_lines)}


@router.get("/{job_id}/frames/{name}")
def frame(job_id: str, name: str, request: Request):
    state = request.app.state
    with errors(), state.runner_lock:
        try:
            content = state.rollout.frame(job_id, name, state.runner)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
