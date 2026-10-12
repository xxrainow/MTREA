"""One holder of the SO-101 arms at a time, across threads and processes.

Two serial clients on one port corrupt each other's packets, and LeRobot does
not guard against it. A session takes this lock before opening a port: a
process-wide threading lock for sessions inside one process (the server), and
data/locks/hardware.lock for other processes (the CLI, experiments/collect.py).
The file holds the owner's pid and process start time; a file whose process is
gone, or whose pid now belongs to another process, is stale and reclaimed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import psutil

from lab.core import paths
from lab.core.hardware import HardwareBusy

# One threading lock per lock file, shared by every HardwareLock in this process.
_PROCESS_LOCKS: dict[Path, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def _process_lock(path: Path) -> threading.Lock:
    with _REGISTRY_LOCK:
        return _PROCESS_LOCKS.setdefault(path, threading.Lock())


def _create_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _alive(info: dict) -> bool:
    pid = info.get("pid")
    if not isinstance(pid, int) or not psutil.pid_exists(pid):
        return False
    started = _create_time(pid)
    recorded = info.get("create_time")
    # A different start time means the pid was reused by an unrelated process.
    if started is not None and isinstance(recorded, (int, float)):
        return abs(started - recorded) < 1.0
    return True


class HardwareLock:
    def __init__(self, path: Path | None = None):
        self.path = (path or paths.hardware_lock()).resolve()
        self._thread_lock = _process_lock(self.path)
        self._held = False
        self.owner: str | None = None

    def _read(self) -> dict | None:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {}  # unreadable: treat as stale, there is no live owner to name
        return info if isinstance(info, dict) else {}

    def holder(self) -> dict | None:
        """Who holds the arms, or None. Stale files count as free."""
        info = self._read()
        if info is None or not _alive(info):
            return None
        return info

    def acquire(self, owner: str) -> None:
        if not self._thread_lock.acquire(blocking=False):
            info = self._read() or {}
            raise HardwareBusy(f"the arms are in use by {info.get('owner', 'another session')} in this process")
        try:
            self._create_file(owner)
        except BaseException:
            self._thread_lock.release()
            raise
        self._held, self.owner = True, owner

    def _create_file(self, owner: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps({
            "pid": os.getpid(),
            "create_time": _create_time(os.getpid()),
            "owner": owner,
            "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = self._read()
                if info is not None and _alive(info) and info.get("pid") != os.getpid():
                    raise HardwareBusy(
                        f"the arms are in use by {info.get('owner', 'another program')} "
                        f"(pid {info.get('pid')}, since {info.get('since', '?')}); "
                        "stop it first") from None
                # Stale, or left by this process without the thread lock: reclaim.
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(record)
            return
        raise HardwareBusy(f"could not take {self.path}; another process is taking it at the same time")

    def release(self) -> None:
        if not self._held:
            return
        try:
            info = self._read()
            if info is not None and info.get("pid") == os.getpid():
                self.path.unlink(missing_ok=True)
        finally:
            self._held, self.owner = False, None
            self._thread_lock.release()

    @property
    def held(self) -> bool:
        return self._held

    def __enter__(self) -> "HardwareLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
