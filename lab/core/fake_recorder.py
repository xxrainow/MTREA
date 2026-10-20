import json
import os
from datetime import datetime
from pathlib import Path
from threading import Event, Lock

from lab.core.dataset_metadata import read_jsonl


def fake_collection_enabled() -> bool:
    return (
        os.environ.get("MTREA_HARDWARE", "real")
        .strip()
        .lower()
        == "fake"
    )


class FakeRecordingProcess:
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

        for _ in range(int(self._args["dataset.num_episodes"])):
            # Accelerated simulation: one episode takes one second.
            if self._stop.wait(1.0):
                break

            episode = {
                "episode_index": len(episodes),
                "tasks": [instruction],
            }
            with episodes_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(episode) + "\n")

            episodes.append(episode)
            self._write_info(info_path, len(episodes))

    @staticmethod
    def _write_info(path: Path, count: int) -> None:
        path.write_text(
            json.dumps({
                "fake": True,
                "total_episodes": count,
            }),
            encoding="utf-8",
        )
