"""Result-list archive actions, independent of rollout control."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from lab.server.rollout_api import errors

router = APIRouter(prefix='/api/rollout-results', tags=['rollout-results'])


class ArchiveBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['validation', 'standalone', 'merged']
    id: str = Field(strict=True, min_length=1)


@router.get('/archive')
def archived(request: Request):
    with errors(), request.app.state.runner_lock:
        return {'archived': request.app.state.rollout_result_archive.archived()}


def change(body, action, request):
    state = request.app.state
    with errors(), state.runner_lock:
        return state.rollout_result_archive.change(body.kind, body.id, action, state.runner)


@router.post('/archive')
def archive(body: ArchiveBody, request: Request):
    return change(body, 'archive', request)


@router.post('/restore')
def restore(body: ArchiveBody, request: Request):
    return change(body, 'restore', request)


class MergeBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sources: list[ArchiveBody] = Field(min_length=2, max_length=50)
    name: str = Field(strict=True, min_length=1, max_length=120)
    request_id: str = Field(strict=True)


@router.get('/merged')
def merged(request: Request):
    with errors(), request.app.state.runner_lock:
        return request.app.state.rollout_result_merge.list(request.app.state.runner)


@router.post('/merge', status_code=201)
def merge(body: MergeBody, request: Request):
    with errors(), request.app.state.runner_lock:
        return request.app.state.rollout_result_merge.create(
            [source.model_dump() for source in body.sources], body.name, body.request_id, request.app.state.runner)
