from fastapi import APIRouter, HTTPException, Request, Path
from fastapi.responses import FileResponse

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
