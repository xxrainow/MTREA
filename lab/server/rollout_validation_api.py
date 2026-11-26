"""Repeated validation settings and explicit per-attempt actions."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from lab.core.hardware import HardwareBusy
from lab.core.models import RolloutConfig
from lab.server.remote_api import get_remote
from lab.server.rollout_api import RolloutBody, errors

router = APIRouter(prefix='/api/rollout-validations', tags=['rollout-validations'])


class ValidationBody(RolloutBody):
    target_count: int = Field(strict=True, ge=1, le=1000)

    def config(self):
        return RolloutConfig(**self.model_dump(exclude={'target_count', 'request_id'}))


class StartBody(ValidationBody):
    request_id: str = Field(pattern='^[a-f0-9]{32}$')


class NextBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: str = Field(pattern='^[a-f0-9]{32}$')
    number: int = Field(strict=True, ge=1, le=1000)
    scene_ready: bool = Field(strict=True)


class OutcomeBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    job_id: str = Field(min_length=1)
    outcome: Literal['success', 'failure', 'unjudged']
    notes: str = Field(default='', max_length=10000)


def devices(state):
    if state.collection.is_active():
        raise HardwareBusy('Stop collection before starting a rollout')
    if state.hardware.backend.name != 'real':
        raise ValueError('Robot rollout is unavailable in fake hardware mode')


@router.post('/check')
def check(body: ValidationBody, request: Request):
    state = request.app.state
    remote, _ = get_remote(request)
    with errors(), state.runner_lock:
        return state.rollout.check(body.config(), state.runner, remote)


@router.post('/start', status_code=201)
def start(body: StartBody, request: Request):
    state = request.app.state
    remote, _ = get_remote(request)
    with errors(), state.collection_gate, state.runner_lock:
        devices(state)
        service = state.rollout_validation
        service.create(body.config(), body.target_count, body.request_id, state.runner, remote)
        return service.next(body.request_id, body.request_id, 1, state.runner, remote, state.cameras.stop)


@router.get('')
def validations(request: Request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout_validation.list(state.runner)


@router.post('/{validation_id}/next', status_code=201)
def next_attempt(validation_id: str, body: NextBody, request: Request):
    state = request.app.state
    remote, _ = get_remote(request)
    with errors(), state.collection_gate, state.runner_lock:
        devices(state)
        if not body.scene_ready:
            raise ValueError('Confirm scene reset and robot readiness before the next attempt')
        return state.rollout_validation.next(validation_id, body.request_id, body.number,
                                              state.runner, remote, state.cameras.stop)


@router.post('/{validation_id}/outcome')
def outcome(validation_id: str, body: OutcomeBody, request: Request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout_validation.annotate(validation_id, **body.model_dump(), runner=state.runner)


@router.post('/{validation_id}/end')
def end(validation_id: str, request: Request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout_validation.end(validation_id, state.runner)
