import json
import os
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from uuid import uuid4

from lab.core.dataset_metadata import read_jsonl
from lab.core.hardware import SessionStateError


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
        self.recording_id = uuid4().hex
        self.started_at = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        self._args = dict(
            item[2:].split("=", 1)
            for item in command[1:]
        )
        self._stop = Event()
        self._advance = Event()
        self._wait_lock = Lock()
        self._exit_code: int | None = None
        self._progress_lock = Lock()
        self._phase = "connecting"
        self._phase_id = uuid4().hex
        self._phase_started = monotonic()
        self._duration = 0.0
        self._phase_active = False
        self._episode = 1
        self._saved = 0
        self._retry_requested = False

    def progress(self) -> dict:
        with self._progress_lock:
            return {
                "episode_phase": self._phase,
                "phase_id": self._phase_id,
                "phase_elapsed_s": min(
                    self._duration, max(0.0, monotonic() - self._phase_started)
                ),
                "phase_duration_s": self._duration or None,
                "current_episode": self._episode,
                "saved_episodes": self._saved,
            }

    def _stage(self, phase: str, duration: float = 0.0) -> bool:
        with self._progress_lock:
            self._advance.clear()
            self._phase = phase
            self._phase_id = uuid4().hex
            self._duration = duration
            self._phase_started = monotonic()
            self._phase_active = duration > 0 and not self._stop.is_set()
            if self._stop.is_set():
                return False

        self._advance.wait(duration)

        with self._progress_lock:
            self._phase_active = False
            return not self._stop.is_set()

    def send_command(
        self,
        action: str,
        *,
        expected_phase_id: str,
        expected_phase: str,
        expected_episode: int,
    ) -> str:
        allowed_actions = {
            "preparing": ("start_episode",),
            "recording": ("finish_episode", "retry_episode"),
            "resetting": ("start_episode",),
        }

        with self._progress_lock:
            if (
                not self._phase_active
                or self._stop.is_set()
                or self._phase != expected_phase
                or self._phase_id != expected_phase_id
                or self._episode != expected_episode
            ):
                raise SessionStateError("Recording phase or episode has changed")

            if action not in allowed_actions.get(self._phase, ()):
                raise SessionStateError("Command is not allowed in this phase")

            if self._advance.is_set():
                raise SessionStateError("A recording command is still pending")

            command_id = uuid4().hex
            self._retry_requested = action == "retry_episode"
            self._advance.set()
            return command_id

    def request_stop(self) -> None:
        with self._progress_lock:
            self._stop.set()
            self._advance.set()

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

        index = 0
        while index < count:
            with self._progress_lock:
                self._episode = index + 1
            if not self._stage(
                "recording", float(self._args["dataset.episode_time_s"])
            ):
                break

            with self._progress_lock:
                retry = self._retry_requested
                self._retry_requested = False

            if index < count - 1 or retry:
                self._stage("resetting", float(self._args["dataset.reset_time_s"]))

            if retry:
                if self._stop.is_set():
                    break
                continue

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
            index += 1
            if self._stop.is_set():
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
