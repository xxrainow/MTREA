import json
import time
from pathlib import Path
from threading import Lock

from lab.core.recording_telemetry import atomic_write


class TrialRecorder:
    def __init__(self, trial_dir: Path):
        self.root = trial_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.events_path = self.root / "events.jsonl"
        self._mutex = Lock()
        self._started = time.monotonic()

        if self.status_path.exists():
            raise FileExistsError(
                f"Trial status already exists: {self.status_path}"
            )

        with self.events_path.open("x", encoding="utf-8"):
            pass

        self.event("created")
        self.write_status(state="connecting")

    def write_status(self, *, state: str, **values) -> None:
        with self._mutex:
            snapshot = {
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
