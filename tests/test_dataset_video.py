import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lab.core.dataset_video import episode_videos, episode_video_file
from lab.server.datasets_api import router


def dataset(tmp_path, version):
    root = tmp_path / "raw" / "demo"
    (root / "meta").mkdir(parents=True)
    key = "observation.images.test_camera"
    info = {"total_episodes": 2, "fps": 30, "chunks_size": 1000,
            "features": {key: {"dtype": "video"}}}
    if version == 3:
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        pd.DataFrame({
            "episode_index": [0, 1],
            f"videos/{key}/chunk_index": [0, 0],
            f"videos/{key}/file_index": [0, 0],
            f"videos/{key}/from_timestamp": [0.0, 2.0],
            f"videos/{key}/to_timestamp": [2.0, 5.0],
        }).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
        video = root / f"videos/{key}/chunk-000/file-000.mp4"
    else:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 1, "length": 90}) + "\n")
        video = root / f"videos/chunk-000/{key}/episode_000001.mp4"
    (root / "meta/info.json").write_text(json.dumps(info))
    video.parent.mkdir(parents=True)
    video.write_bytes(b"0123456789abcdef")
    return root, video


def test_v3_episodes_resolve_shared_file_with_distinct_times(tmp_path):
    _, file = dataset(tmp_path, 3)
    first = episode_videos("real", "demo", 0, tmp_path)[0]
    second = episode_videos("real", "demo", 1, tmp_path)[0]
    assert (first.start_s, first.end_s) == (0, 2)
    assert (second.start_s, second.end_s) == (2, 5)
    assert episode_video_file("real", "demo", 1, second.camera, tmp_path) == file


def test_v2_video_uses_episode_filename_and_duration(tmp_path):
    _, file = dataset(tmp_path, 2)
    clip = episode_videos("real", "demo", 1, tmp_path)[0]
    assert (clip.start_s, clip.end_s) == (0, 3)
    assert episode_video_file("real", "demo", 1, clip.camera, tmp_path) == file
    with pytest.raises(FileNotFoundError):
        episode_videos("real", "demo", 0, tmp_path)


def test_stream_supports_ranges_head_and_unknown_camera(tmp_path):
    dataset(tmp_path, 2)
    app = FastAPI()
    app.state.data_base = tmp_path
    app.include_router(router)
    with TestClient(app) as client:
        base = "/api/datasets/real/demo/episodes/1/videos"
        assert client.get(base).json()[0]["end_s"] == 3
        response = client.get(f"{base}/test_camera", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.headers["content-range"] == "bytes 2-5/16"
        assert response.content == b"2345"
        assert response.headers["content-type"] == "video/mp4"
        head = client.head(f"{base}/test_camera")
        assert head.status_code == 200
        assert head.content == b""
        assert client.get(f"{base}/unknown").status_code == 404
        assert client.get("/api/datasets/real/demo/episodes/-1/videos").status_code == 422


def test_external_video_symlink_and_metadata_path_are_rejected(tmp_path):
    root, file = dataset(tmp_path, 2)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private")
    file.unlink()
    file.symlink_to(outside)
    with pytest.raises(ValueError):
        episode_video_file("real", "demo", 1, "test_camera", tmp_path)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = "../outside.mp4"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        episode_videos("real", "demo", 1, tmp_path)


def test_missing_video_and_fake_video(tmp_path):
    _, file = dataset(tmp_path, 2)
    file.unlink()
    with pytest.raises(FileNotFoundError):
        episode_video_file("real", "demo", 1, "test_camera", tmp_path)
    fake_root = tmp_path / "fake/collection/raw/fake_demo/meta"
    fake_root.mkdir(parents=True)
    assert episode_videos("fake", "fake_demo", 0, tmp_path) == []
