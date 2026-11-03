from fastapi import APIRouter, HTTPException, Request, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from lab.core.dataset_edit import check_edit, edit_command
from lab.core.dataset_merge import check_merge, merge_command
from lab.core.dataset_manifest import ManifestError

from lab.core.datasets import Backend, DatasetDetail, DatasetSummary, get_dataset, list_datasets
from lab.core.dataset_video import EpisodeVideo, episode_videos, episode_video_file
from lab.core.dataset_joints import read_episode_joints
from lab.core.datasets import dataset_root
from lab.core.dataset_delete import archive_dataset
from lab.core.hardware import HardwareBusy
from lab.core.dataset_sync import remote_state, sendability, sync_command
from lab.core.models import JobKind, JobSpec
from lab.core.runner import JobConflict
from lab.server.remote_api import get_remote


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetSummary])
def dataset_list(request: Request) -> list[DatasetSummary]:
    return list_datasets(request.app.state.data_base)


class MergeRequest(BaseModel):
    backend: Backend
    sources: list[str] = Field(min_length=2, max_length=50)
    name: str
    source_versions: list[str] | None = None


def _check_merge(body: MergeRequest, request: Request) -> dict:
    try:
        checked = check_merge(body.backend, body.sources, body.name, request.app.state.data_base)
        if body.source_versions is not None and body.source_versions != checked["source_versions"]:
            raise ValueError("Source datasets changed after review; check the selection again")
        return checked
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (ValueError, ManifestError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (OSError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=f"Cannot check merge: {exc}") from None


@router.post("/merge/check")
def dataset_merge_check(body: MergeRequest, request: Request) -> dict:
    return _check_merge(body, request)


@router.post("/merge", status_code=202)
def dataset_merge(body: MergeRequest, request: Request) -> dict:
    state = request.app.state
    with state.collection_gate:
        if state.collection.is_active():
            raise HTTPException(status_code=409, detail="Stop collection before merging datasets")
        if body.source_versions is None:
            raise HTTPException(status_code=422, detail="Check the merge before submitting")
        checked = _check_merge(body, request)
        spec = JobSpec(kind=JobKind.MERGE, cmd=merge_command(checked, state.data_base),
                       label=f"merge {body.backend}/{body.name}")
        with state.runner_lock:
            try:
                job = state.runner.submit(spec)
            except JobConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None
        return {"job": job.to_dict(), "output": {"backend": body.backend, "name": body.name}}


@router.get("/{backend}/{name}", response_model=DatasetDetail)
def dataset_detail(backend: Backend, name: str, request: Request) -> DatasetDetail:
    try:
        return get_dataset(backend, name, request.app.state.data_base)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None


def _video_read(reader, *args):
    try:
        return reader(*args)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read episode video metadata: {exc}") from None
    except (OSError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=f"Video unavailable: {exc}") from None


@router.get("/{backend}/{name}/episodes/{episode_index}/videos", response_model=list[EpisodeVideo])
def video_list(backend: Backend, name: str, request: Request, episode_index: int = Path(ge=0)):
    return _video_read(episode_videos, backend, name, episode_index, request.app.state.data_base)


@router.api_route("/{backend}/{name}/episodes/{episode_index}/videos/{camera}", methods=["GET", "HEAD"])
def video_file(backend: Backend, name: str, camera: str, request: Request, episode_index: int = Path(ge=0)):
    path = _video_read(episode_video_file, backend, name, episode_index, camera, request.app.state.data_base)
    return FileResponse(path, media_type="video/mp4")


@router.get(
    "/{backend}/{name}/episodes/{episode_index}/joints",
)
def episode_joints(
    backend: Backend,
    name: str,
    request: Request,
    episode_index: int = Path(ge=0),
) -> dict:
    try:
        root = dataset_root(
            backend,
            name,
            request.app.state.data_base,
        )
        return read_episode_joints(root, episode_index)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from None

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from None

    except (OSError, ImportError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Joint data unavailable: {exc}",
        ) from None


@router.delete("/{backend}/{name}")
def delete_dataset(
    backend: Backend,
    name: str,
    request: Request,
) -> dict:
    try:
        with request.app.state.collection_gate:
            if request.app.state.collection.is_active():
                raise HTTPException(
                    status_code=409,
                    detail="Stop collection before deleting a dataset",
                )

            return archive_dataset(
                backend,
                name,
                request.app.state.data_base,
            )

    except HardwareBusy as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from None

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from None

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from None

    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot archive dataset: {exc}",
        ) from None


def _existing_dataset(backend: Backend, name: str, request: Request) -> None:
    try:
        dataset_root(backend, name, request.app.state.data_base)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Dataset not found") from None


@router.get("/{backend}/{name}/remote")
def dataset_remote(backend: Backend, name: str, request: Request) -> dict:
    """Whether the GPU host holds this dataset, and where. remote_path is what training uses."""
    _existing_dataset(backend, name, request)
    remote, why = get_remote(request)
    return remote_state(remote, backend, name, request.app.state.data_base, not_configured=why)


@router.post("/{backend}/{name}/sync")
def dataset_sync(backend: Backend, name: str, request: Request) -> dict:
    _existing_dataset(backend, name, request)
    state = request.app.state
    send = sendability(backend, name, state.data_base)
    if not send.sendable:
        raise HTTPException(status_code=409, detail=f"Not sendable: {send.reason}")
    remote, why = get_remote(request)
    if remote is None:
        raise HTTPException(status_code=409, detail=f"GPU host not configured: {why}")
    spec = JobSpec(
        kind=JobKind.SYNC,
        cmd=sync_command(name, state.data_base, state.remote_config, state.remote_transport),
        label=f"sync {name}",
    )
    with state.runner_lock:
        try:
            job = state.runner.submit(spec)
        except JobConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
    state.remote_cache.clear()
    return job.to_dict()


class DatasetEditRequest(BaseModel):
    name: str
    instructions: dict[str, str]
    source_version: str | None = None


def _check_edit(backend: Backend, source: str, body: DatasetEditRequest, request: Request) -> dict:
    try:
        checked = check_edit(backend, source, body.name, body.instructions, request.app.state.data_base)
        if body.source_version is not None and body.source_version != checked["source_version"]:
            raise ValueError("Dataset changed after review; reload and check again")
        return checked
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (ValueError, ManifestError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (OSError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=f"Cannot check dataset edit: {exc}") from None


@router.post("/{backend}/{name}/edit/check")
def dataset_edit_check(backend: Backend, name: str, body: DatasetEditRequest, request: Request) -> dict:
    return _check_edit(backend, name, body, request)


@router.post("/{backend}/{name}/edit", status_code=202)
def dataset_edit(backend: Backend, name: str, body: DatasetEditRequest, request: Request) -> dict:
    state = request.app.state
    with state.collection_gate:
        if state.collection.is_active():
            raise HTTPException(status_code=409, detail="Stop collection before editing datasets")
        if body.source_version is None:
            raise HTTPException(status_code=422, detail="Check the edit before submitting")
        checked = _check_edit(backend, name, body, request)
        with state.runner_lock:
            try:
                spec = JobSpec(kind=JobKind.EDIT, cmd=edit_command(checked, state.data_base),
                               label=f"edit {backend}/{body.name}")
                job = state.runner.submit(spec)
            except JobConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=503, detail=f"Cannot start dataset edit: {exc}") from None
        return {"job": job.to_dict(), "output": {"backend": backend, "name": body.name}}
