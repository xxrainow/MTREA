"""HTTP routes for camera preview."""

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from lab.core.hardware import HardwareError
from lab.core.hardware.cameras import CameraService
from lab.core.fake_recorder import fake_collection_enabled

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


def _service(request: Request) -> CameraService:
    return request.app.state.cameras


def _call(fn, *args):
    try:
        return fn(*args)
    except HardwareError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("/config")
def camera_config(request: Request) -> list[dict]:
    return _call(_service(request).config)


@router.get("/status")
def camera_status(request: Request) -> dict:
    return _service(request).status()


@router.post("/start")
def camera_start(request: Request) -> dict:
    with request.app.state.collection_gate:
        if request.app.state.collection.is_active():
            raise HTTPException(
                status_code=409,
                detail="Camera preview is unavailable during collection",
            )
    with request.app.state.collection_gate:
        if fake_collection_enabled():
            raise HTTPException(
                status_code=409,
                detail="Camera preview is unavailable in fake mode",
            )

        return _call(_service(request).start)


@router.post("/stop")
def camera_stop(request: Request) -> dict:
    return _service(request).stop()


@router.get("/{name}/stream")
def camera_stream(name: str, request: Request):
    service = _service(request)
    configs = _call(service.config)
    config = next((item for item in configs if item["name"] == name), None)

    if config is None:
        raise HTTPException(status_code=404, detail="Unknown camera")

    if name not in service.status()["cameras"]:
        raise HTTPException(status_code=409, detail="Start preview first")

    first_frame = _call(service.read_jpeg, name)

    def frames():
        jpeg = first_frame
        while True:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

            time.sleep(1 / config["fps"])
            try:
                jpeg = service.read_jpeg(name)
            except HardwareError:
                break

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )
