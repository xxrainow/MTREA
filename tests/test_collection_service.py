import json
from pathlib import Path

import pytest

import lab.core.collection as collection_module
from lab.core.collection import CollectionService
from lab.core.collection_ledger import read_ledger
from lab.core.models import CollectionPhase


class FakeRecording:
    commands = []

    def __init__(self, command):
        self.args = dict(
            item[2:].split("=", 1)
            for item in command[1:]
        )
        self.started_at = "2026-10-20T10:00:00+09:00"
        self._done = False
        self.commands.append(self.args)

    def wait(self):
        if self._done:
            return 0

        root = Path(self.args["dataset.root"])
        meta = root / "meta"
        meta.mkdir(parents=True, exist_ok=True)

        episodes_path = meta / "episodes.jsonl"
        episodes = (
            [
                json.loads(line)
                for line in episodes_path.read_text().splitlines()
            ]
            if episodes_path.exists()
            else []
        )

        instruction = self.args["dataset.single_task"]
        count = int(self.args["dataset.num_episodes"])
        start = len(episodes)

        episodes.extend(
            {
                "episode_index": start + offset,
                "tasks": [instruction],
            }
            for offset in range(count)
        )

        episodes_path.write_text(
            "".join(json.dumps(row) + "\n" for row in episodes),
            encoding="utf-8",
        )
        tasks = sorted({
            task
            for episode in episodes
            for task in episode["tasks"]
        })
        (meta / "tasks.jsonl").write_text(
            "".join(
                json.dumps({"task": task}) + "\n"
                for task in tasks
            ),
            encoding="utf-8",
        )
        (meta / "info.json").write_text(
            json.dumps({"total_episodes": len(episodes)}),
            encoding="utf-8",
        )

        self._done = True
        return 0


def test_records_two_sessions_and_rejects_duplicate(
    tmp_path, monkeypatch
):
    FakeRecording.commands = []
    monkeypatch.setattr(
        collection_module, "RecordingProcess", FakeRecording
    )
    monkeypatch.setattr(
        collection_module, "build_hardware_args", lambda: []
    )

    service = CollectionService(data_base=tmp_path)
    root = tmp_path / "raw" / "demo"
    ledger = tmp_path / "raw" / "demo.collection.jsonl"

    def start_session(index):
        service.start(
            dataset_root=root,
            ledger_path=ledger,
            plan_sha="fake-plan-hash",
            session_index=index,
            planned_episodes=2,
            repo_id="local/demo",
            instruction=f"session {index}",
            episode_time_s=20,
            reset_time_s=15,
            record_context={
                "object": f"object {index}",
                "plan": "fake-plan",
                "git_commit": "fake-commit",
                "git_dirty": False,
                "lerobot_version": "fake",
                "layouts": [
                    {"target_position": "P1"},
                    {"target_position": "P2"},
                ],
            },
        )
        service._worker.join(timeout=3)
        assert not service._worker.is_alive()
        assert service.status().phase == CollectionPhase.FINISHED, status.error

    start_session(1)
    start_session(2)

    records = read_ledger(ledger)
    assert [row["session"] for row in records] == [1, 2]
    assert [row["episodes_before"] for row in records] == [0, 2]
    assert [row["episodes_after"] for row in records] == [2, 4]
    assert [
        args["resume"] for args in FakeRecording.commands
    ] == ["false", "true"]
    assert [
        item["episode_index"]
        for row in records
        for item in row["layouts"]
    ] == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="cannot be recorded again"):
        start_session(1)

    assert len(read_ledger(ledger)) == 2
    assert service._hardware_lock.holder() is None
