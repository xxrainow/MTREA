import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock

from lab.core.recording_telemetry import atomic_write


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def build_trial_metadata(config, *, time_limit_s: float,
                         repo_root: Path | None = None,
                         task: str | None = None) -> dict:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]

    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True,
                check=True, timeout=5,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain") if head else None
    try:
        lerobot_version = version("lerobot")
    except PackageNotFoundError:
        lerobot_version = None
    return {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": head or None,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "lerobot_version": lerobot_version,
        "checkpoint": config.pretrained_name_or_path,
        "server_address": config.server_address,
        "task": config.task if task is None else task,
        "robot_id": config.robot.id,
        "time_limit_s": time_limit_s,
        "configuration": asdict(config),
    }


class TrialRecorder:
    def __init__(self, trial_dir: Path):
        self.root = trial_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.events_path = self.root / "events.jsonl"
        self.session_path = self.root / "session.json"
        self._mutex = Lock()
        self._started = time.monotonic()
        self._context = {}

        if self.status_path.exists() or self.session_path.exists():
            raise FileExistsError(
                f"Trial status already exists: {self.status_path}"
            )

        with self.events_path.open("x", encoding="utf-8"):
            pass

        self.event("created")
        self.write_status(state="connecting")

    def write_session(self, metadata: dict) -> None:
        with self._mutex:
            if self.session_path.exists():
                raise FileExistsError("Trial session metadata is immutable")
            content = json.dumps(
                metadata, ensure_ascii=False, allow_nan=False, default=_json_default,
            ).encode("utf-8")
            atomic_write(self.session_path, content)

    def write_status(self, *, state: str, **values) -> None:
        with self._mutex:
            snapshot = {
                **self._context,
                **values,
                "state": state,
                "elapsed_s": time.monotonic() - self._started,
            }
            content = json.dumps(
                snapshot,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            atomic_write(self.status_path, content)

    def set_context(self, **values) -> None:
        with self._mutex:
            self._context.update(values)

    def update_status(self, **values) -> None:
        with self._mutex:
            snapshot = json.loads(self.status_path.read_text(encoding="utf-8"))
            snapshot.update(self._context)
            snapshot.update(values)
            atomic_write(self.status_path, json.dumps(
                snapshot, ensure_ascii=False, allow_nan=False,
            ).encode("utf-8"))

    def event(self, kind: str, **details) -> None:
        with self._mutex:
            record = {
                "event": kind,
                "timestamp": time.time(),
                "elapsed_s": time.monotonic() - self._started,
                "details": details,
            }
            line = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            )
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
