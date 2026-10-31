"""GPU host status and dataset sync. Delegates to lab.core.remote and lab.core.dataset_sync."""

import time
from threading import Lock
from typing import Callable

from fastapi import APIRouter, Request

from lab.core.dataset_sync import remote_markers
from lab.core.remote import Remote, RemoteNotConfigured, load_remote

router = APIRouter(prefix="/api/remote", tags=["remote"])

CACHE_S = 5.0


class TtlCache:
    """One value per key, recomputed after CACHE_S. Keeps page polling from opening an SSH session each time."""

    def __init__(self, ttl_s: float = CACHE_S):
        self.ttl_s = ttl_s
        self._lock = Lock()
        self._values: dict[str, tuple[float, object]] = {}

    def get(self, key: str, compute: Callable[[], object]):
        with self._lock:
            hit = self._values.get(key)
            if hit is not None and time.monotonic() - hit[0] < self.ttl_s:
                return hit[1]
        value = compute()
        with self._lock:
            self._values[key] = (time.monotonic(), value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


def get_remote(request: Request) -> tuple[Remote | None, str]:
    """(remote, "") or (None, why it is not configured)."""
    state = request.app.state
    try:
        return load_remote(state.remote_config, state.remote_transport), ""
    except RemoteNotConfigured as e:
        return None, str(e)


@router.get("/status")
def remote_status(request: Request) -> dict:
    def compute() -> dict:
        remote, why = get_remote(request)
        if remote is None:
            return {"configured": False, "reachable": False, "message": why, "host": None}
        ok, message = remote.ping(timeout=6)
        return {"configured": True, "reachable": ok, "message": message, "host": remote.label}

    return request.app.state.remote_cache.get("status", compute)


@router.get("/datasets")
def dataset_markers(request: Request) -> dict[str, str]:
    """dataset id -> synced | different | not_synced | not_sent_fake | unknown, for the list view."""
    def compute() -> dict[str, str]:
        remote, _ = get_remote(request)
        return remote_markers(remote, request.app.state.data_base)

    return request.app.state.remote_cache.get("markers", compute)
