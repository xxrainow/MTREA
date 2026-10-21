import json
import os
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from time import monotonic

from lab.core.dataset_metadata import read_jsonl


def fake_collection_enabled() -> bool:
    return (
        os.environ.get("MTREA_HARDWARE", "real")
        .strip()
        .lower()
        == "fake"
    )


class FakeRecordingProcess:
    PREPARATION_SECONDS = 3.0

    def __init__(self, command: list[str]) -> None:
        self.started_at = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        self._args = dict(
            item[2:].split("=", 1)
            for item in command[1:]
        )
        self._stop = Event()
        self._wait_lock = Lock()
        self._exit_code: int | None = None
        self._progress_lock = Lock()
        self._phase = "preparing"
        self._phase_started = monotonic()
        self._duration = self.PREPARATION_SECONDS
        self._episode = 1
        self._saved = 0

    def progress(self) -> dict:
        with self._progress_lock:
            return {
                "episode_phase": self._phase,
                "phase_elapsed_s": min(
                    self._duration, max(0.0, monotonic() - self._phase_started)
                ),
                "phase_duration_s": self._duration or None,
                "current_episode": self._episode,
                "saved_episodes": self._saved,
            }

    def _stage(self, phase: str, duration: float = 0.0) -> bool:
        with self._progress_lock:
            self._phase = phase
            self._duration = duration
            self._phase_started = monotonic()
        return not self._stop.wait(duration)

    def request_stop(self) -> None:
        self._stop.set()

    def wait(self) -> int:
        with self._wait_lock:
            if self._exit_code is not None:
                return self._exit_code

            try:
                self._record()
            except Exception:
                self._exit_code = 1
                raise

            self._stage("stopped" if self._stop.is_set() else "completed")
            self._exit_code = 0
            return self._exit_code

    def _record(self) -> None:
        root = Path(self._args["dataset.root"])
        meta = root / "meta"
        info_path = meta / "info.json"
        episodes_path = meta / "episodes.jsonl"

        if self._args["resume"] == "true":
            info = json.loads(
                info_path.read_text(encoding="utf-8")
            )
            if info.get("fake") is not True:
                raise ValueError("Cannot resume a real dataset in fake mode")

            episodes = read_jsonl(episodes_path)
        else:
            root.mkdir(parents=True, exist_ok=False)
            meta.mkdir()
            episodes = []

        instruction = self._args["dataset.single_task"]
        tasks = sorted(
            {
                task
                for episode in episodes
                for task in episode["tasks"]
            }
            | {instruction}
        )
        (meta / "tasks.jsonl").write_text(
            "".join(
                json.dumps({"task": task}) + "\n"
                for task in tasks
            ),
            encoding="utf-8",
        )
        episodes_path.touch(exist_ok=True)
        self._write_info(info_path, len(episodes))

        count = int(self._args["dataset.num_episodes"])
        if not self._stage("preparing", self.PREPARATION_SECONDS):
            return

        for index in range(count):
            with self._progress_lock:
                self._episode = index + 1
            if not self._stage(
                "recording", float(self._args["dataset.episode_time_s"])
            ):
                break

            self._stage("saving")
            episode = {
                "episode_index": len(episodes),
                "tasks": [instruction],
            }
            with episodes_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(episode) + "\n")

            episodes.append(episode)
            self._write_info(info_path, len(episodes))
            with self._progress_lock:
                self._saved += 1
            if index < count - 1 and not self._stage(
                "resetting", float(self._args["dataset.reset_time_s"])
            ):
                break

    @staticmethod
    def _write_info(path: Path, count: int) -> None:
        path.write_text(
            json.dumps({
                "fake": True,
                "total_episodes": count,
            }),
            encoding="utf-8",
        )
