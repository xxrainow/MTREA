import json

import pytest
from fastapi.testclient import TestClient

from lab.core.datasets import get_dataset, list_datasets
from lab.core.fake_recorder import FakeRecordingProcess
from lab.server.main import create_app


def write_dataset(root, count=2):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "total_episodes": count, "total_frames": count * 30,
        "fps": 30, "robot_type": "so101_follower",
        "features": {"observation.images.test_camera": {"dtype": "video"}},
    }))
    (meta / "tasks.jsonl").write_text(json.dumps({"task": "Move the block."}) + "\n")
    (meta / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": index, "tasks": ["Move the block."]}) + "\n"
        for index in range(count)
    ))


def test_catalog_separates_fake_real_and_ignores_config_files(tmp_path):
    write_dataset(tmp_path / "raw" / "demo")
    write_dataset(tmp_path / "fake" / "collection" / "raw" / "demo", count=1)
    (tmp_path / "raw" / "only.collection.input.json").write_text("{}")
    rows = list_datasets(tmp_path)
    assert [(row.id, row.episodes) for row in rows] == [("real/demo", 2), ("fake/demo", 1)]
    detail = get_dataset("real", "demo", tmp_path)
    assert detail.dataset.cameras == ["test_camera"]
    assert detail.dataset.location == "raw/demo"
    assert detail.dataset.total_frames == 60
    assert detail.tasks == ["Move the block."]
    assert [episode.index for episode in detail.episodes] == [0, 1]


def test_bad_metadata_does_not_hide_healthy_dataset(tmp_path):
    write_dataset(tmp_path / "raw" / "healthy")
    write_dataset(tmp_path / "raw" / "broken")
    (tmp_path / "raw" / "broken" / "meta" / "info.json").write_text("{")
    rows = {row.name: row for row in list_datasets(tmp_path)}
    assert rows["healthy"].episodes == 2
    assert rows["healthy"].issues == []
    assert rows["broken"].episodes is None
    assert rows["broken"].issues


def test_external_dataset_and_metadata_symlinks_are_not_read(tmp_path):
    outside = tmp_path / "outside"
    base = tmp_path / "data"
    write_dataset(outside)
    (base / "raw").mkdir(parents=True)
    (base / "raw" / "linked").symlink_to(outside, target_is_directory=True)
    assert list_datasets(base) == []
    with pytest.raises(FileNotFoundError):
        get_dataset("real", "linked", base)
    (base / "raw" / "nested").mkdir()
    (base / "raw" / "nested" / "meta").symlink_to(outside / "meta", target_is_directory=True)
    detail = get_dataset("real", "nested", base)
    assert detail.episodes == []
    assert detail.dataset.issues
    with pytest.raises(FileNotFoundError):
        get_dataset("real", "../outside", base)


def test_fake_collection_appears_in_catalog_and_detail(tmp_path, monkeypatch):
    monkeypatch.setenv("MTREA_HARDWARE", "fake")
    monkeypatch.setattr(FakeRecordingProcess, "PREPARATION_SECONDS", 0.01)
    app = create_app(data_base=tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/datasets").json() == []
        started = client.post("/api/collection/start", json={
            "dataset": "web_demo", "session_index": 1,
            "sessions": [{"instruction": "Move the block.", "num_episodes": 2,
                          "episode_time_s": 0.01, "reset_time_s": 0.01}],
        })
        assert started.status_code == 202, started.text
        app.state.collection._worker.join(timeout=3)
        assert not app.state.collection._worker.is_alive()
        assert app.state.collection.status().error is None
        listing = client.get("/api/datasets")
        assert listing.status_code == 200
        assert listing.json()[0]["id"] == "fake/web_demo"
        assert listing.json()[0]["episodes"] == 2
        response = client.get("/api/datasets/fake/web_demo")
        assert response.status_code == 200, response.text
        detail = response.json()
        assert len(detail["episodes"]) == 2
        assert detail["episodes"][1]["tasks"] == ["Move the block."]
        assert detail["dataset"]["total_frames"] is None
        assert detail["dataset"]["issues"] == []
        assert client.get("/api/datasets/real/web_demo").status_code == 404


def test_v3_episode_metadata(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path / "raw" / "v3_demo"
    meta = root / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))
    pd.DataFrame({"task": ["Move the block."]}).to_parquet(meta / "tasks.parquet")
    pd.DataFrame({"episode_index": [0], "tasks": [["Move the block."]]}).to_parquet(
        meta / "episodes" / "chunk-000" / "file-000.parquet"
    )
    detail = get_dataset("real", "v3_demo", tmp_path)
    assert detail.dataset.issues == []
    assert detail.episodes[0].tasks == ["Move the block."]
