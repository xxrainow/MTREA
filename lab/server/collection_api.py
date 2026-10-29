from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from lab.core.collection import CollectionService
from lab.core.models import CollectionStatus
from lab.core.collection_input import prepare_start_arguments
from lab.core.hardware import (
    HardwareBusy,
    HardwareError,
    SessionStateError,
)
from lab.server.collection_schemas import (
    CollectionCommandRequest,
    CollectionCommandResponse,
    CollectionInputRequest,
    CollectionStartRequest,
)


router = APIRouter(
    prefix="/api/collection",
    tags=["collection"],
)


@router.get("/status", response_model=CollectionStatus)
def collection_status(request: Request) -> CollectionStatus:
    service: CollectionService = request.app.state.collection
    return service.status()


@router.get("/frames/{recording_id}/{name}")
def recording_frame(recording_id: str, name: str, request: Request):
    service: CollectionService = request.app.state.collection
    try:
        jpeg = service.read_frame(recording_id, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except OSError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

@router.post(
    "/validate",
    response_model=CollectionInputRequest,
)
def validate_collection_input(
    body: CollectionInputRequest,
) -> CollectionInputRequest:
    return body



@router.post(
    "/start",
    response_model=CollectionStatus,
    status_code=202,
)
def start_collection(
    body: CollectionStartRequest,
    request: Request,
) -> CollectionStatus:
    service: CollectionService = request.app.state.collection

    try:
        with request.app.state.collection_gate:
            if service.is_active():
                raise SessionStateError(
                    "Collection is already active"
                )

            arguments = prepare_start_arguments(
                body.model_dump(),
                data_base=request.app.state.data_base,
            )

            request.app.state.cameras.stop()
            return service.start(**arguments)

    except (HardwareBusy, SessionStateError) as exc:
        raise HTTPException(
            status_code=409, detail=str(exc)
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from None
    except (HardwareError, OSError) as exc:
        raise HTTPException(
            status_code=503, detail=str(exc)
        ) from None


@router.post("/stop", response_model=CollectionStatus)
def stop_collection(request: Request) -> CollectionStatus:
    service: CollectionService = request.app.state.collection

    try:
        return service.stop()
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=str(exc)
        ) from None


@router.post(
    "/command",
    response_model=CollectionCommandResponse,
    status_code=202,
)
def send_collection_command(
    body: CollectionCommandRequest,
    request: Request,
) -> CollectionCommandResponse:
    service: CollectionService = request.app.state.collection

    try:
        command_id = service.send_command(
            **body.model_dump()
        )

        return CollectionCommandResponse(
            command_id=command_id,
            status="queued",
        )

    except SessionStateError as exc:
        raise HTTPException(
            status_code=409,
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
            detail=str(exc),
        ) from None
