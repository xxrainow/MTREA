from fastapi import APIRouter, HTTPException, Request, Path
from fastapi.responses import FileResponse

from lab.core.datasets import Backend, DatasetDetail, DatasetSummary, get_dataset, list_datasets
from lab.core.dataset_video import EpisodeVideo, episode_videos, episode_video_file


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
