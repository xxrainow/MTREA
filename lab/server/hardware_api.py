"""HTTP and WebSocket routes for lab/core/hardware. No robot code here.

Handlers validate the request and call app.state.hardware (a core
HardwareService). They are plain `def`s, so FastAPI runs them in its
threadpool: port detection waits up to 15 s and teleop start opens two serial
ports. Errors become HTTP status codes with the core message as `detail`:
409 lock held or wrong state, 422 port detection failed, 503 other hardware
failures (port won't open, missing calibration).
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from lab.core.hardware import HardwareBusy, HardwareError, PortDetectionError, SessionStateError
from lab.core.hardware.service import HardwareService

router = APIRouter()

Arm = Literal["follower", "leader"]


class ArmBody(BaseModel):
    arm: Arm


class PortBody(BaseModel):
    arm: Arm
    port: str = Field(min_length=1)


def _hardware(request: Request) -> HardwareService:
    return request.app.state.hardware


def _call(fn, *args):
    try:
        return fn(*args)
    except (HardwareBusy, SessionStateError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except PortDetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except HardwareError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


# --- config and ports ---

@router.get("/api/hardware/config")
def hardware_config(request: Request) -> dict:
    return _call(_hardware(request).config)


@router.get("/api/hardware/ports")
def hardware_ports(request: Request) -> dict:
    return _call(_hardware(request).ports)


@router.post("/api/hardware/ports/detect/start")
def detect_start(body: ArmBody, request: Request) -> dict:
    return _call(_hardware(request).detect_start, body.arm)


@router.post("/api/hardware/ports/detect/finish")
def detect_finish(body: ArmBody, request: Request) -> dict:
    return _call(_hardware(request).detect_finish, body.arm)


@router.post("/api/hardware/ports/save")
def save_port(body: PortBody, request: Request) -> dict:
    return _call(_hardware(request).save_port, body.arm, body.port)


# --- calibration ---

@router.post("/api/calibration/start")
def calibration_start(body: ArmBody, request: Request) -> dict:
    return _call(_hardware(request).calibration.start, body.arm)


@router.post("/api/calibration/next")
def calibration_next(request: Request) -> dict:
    return _call(_hardware(request).calibration.next)


@router.post("/api/calibration/cancel")
def calibration_cancel(request: Request) -> dict:
    return _call(_hardware(request).calibration.cancel)


@router.get("/api/calibration/status")
def calibration_status(request: Request) -> dict:
    return _hardware(request).calibration.status()


# --- teleoperation ---

@router.post("/api/teleop/start")
def teleop_start(request: Request) -> dict:
    return _call(_hardware(request).teleop.start)


@router.post("/api/teleop/stop")
def teleop_stop(request: Request) -> dict:
    return _call(_hardware(request).teleop.stop)


@router.get("/api/teleop/status")
def teleop_status(request: Request) -> dict:
    return _hardware(request).teleop.status()


# --- joint stream ---

@router.websocket("/ws/joints")
async def joints(websocket: WebSocket) -> None:
    """Joint frames from teleop and calibration. Latest wins: a slow client skips frames."""
    await websocket.accept()
    loop = asyncio.get_running_loop()
    latest: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)

    def offer(frame: dict) -> None:  # on the event loop
        if latest.full():
            latest.get_nowait()
        latest.put_nowait(frame)

    def push(frame: dict) -> None:  # on the session's worker thread; must not block
        loop.call_soon_threadsafe(offer, frame)

    unsubscribe = websocket.app.state.hardware.stream.subscribe(push)

    async def send() -> None:
        while True:
            await websocket.send_json(await latest.get())

    async def receive() -> None:
        while True:
            await websocket.receive_text()  # raises once the client goes away

    tasks = {asyncio.create_task(send()), asyncio.create_task(receive())}
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
        for task in tasks:
            task.cancel()
