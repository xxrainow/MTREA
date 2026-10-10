import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robot.config import CAMERAS, FPS, JOINT_ORDER


spec = importlib.util.spec_from_file_location(
    "inspect_dataset", Path(__file__).resolve().parents[1] / "experiments/inspect_dataset.py"
)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


@pytest.fixture
def dataset(tmp_path):
    features = {
        key: {"dtype": "float32", "shape": [len(JOINT_ORDER)],
              "names": [f"{name}.pos" for name in JOINT_ORDER]}
        for key in ("observation.state", "action")
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera.name}"] = {
            "dtype": "video", "shape": [camera.height, camera.width, 3]
        }
        directory = tmp_path / "videos" / f"observation.images.{camera.name}"
        directory.mkdir(parents=True)
        (directory / "file-000.mp4").write_bytes(b"not decoded in basic checks")
    (tmp_path / "meta").mkdir()
    info = {"fps": FPS, "total_episodes": 2, "total_frames": 6, "features": features}
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    rows = [
        {"episode_index": episode, "frame_index": frame, "timestamp": frame / FPS,
         "observation.state": np.zeros(len(JOINT_ORDER)),
         "action": np.ones(len(JOINT_ORDER))}
        for episode in range(2) for frame in range(3)
    ]
    (tmp_path / "data").mkdir()
    # Both an episode boundary in a file and an episode split across files.
    pd.DataFrame(rows[:4]).to_parquet(tmp_path / "data/file-000.parquet")
    pd.DataFrame(rows[4:]).to_parquet(tmp_path / "data/file-001.parquet")
    return tmp_path


def change_second_file(root, change):
    path = root / "data/file-001.parquet"
    df = pd.read_parquet(path)
    change(df)
    df.to_parquet(path)


def test_valid_split_episodes_and_timestamp_reset(dataset, capsys):
    assert inspector.main(dataset) == 0
    output = capsys.readouterr().out
    assert "video decoding NOT checked" in output
    assert "implemented checks passed" in output


@pytest.mark.parametrize("value", [np.full(len(JOINT_ORDER), np.nan),
                                       np.full(len(JOINT_ORDER), np.inf),
                                       np.zeros(len(JOINT_ORDER) - 1)])
def test_invalid_vector_in_later_file(dataset, capsys, value):
    change_second_file(dataset, lambda df: df.at.__setitem__((0, "action"), value))
    assert inspector.main(dataset) == 2
    assert "file-001.parquet: action: 1 invalid rows" in capsys.readouterr().out


@pytest.mark.parametrize("timestamps", [[0, 0], [0.2, 0.1], [1 / FPS, 1.0]])
def test_bad_episode_timestamps(dataset, capsys, timestamps):
    change_second_file(dataset, lambda df: df.__setitem__("timestamp", timestamps))
    assert inspector.main(dataset) == 2
    assert "episode 1:" in capsys.readouterr().out


def test_duplicate_frame_across_files(dataset, capsys):
    change_second_file(dataset, lambda df: df.__setitem__("frame_index", [0, 2]))
    assert inspector.main(dataset) == 2
    assert "without gaps or duplicates" in capsys.readouterr().out


def test_missing_required_column(dataset, capsys):
    change_second_file(dataset, lambda df: df.drop(columns=["action"], inplace=True))
    assert inspector.main(dataset) == 2
    assert "missing columns ['action']" in capsys.readouterr().out


def test_corrupt_parquet(dataset, capsys):
    (dataset / "data/file-001.parquet").write_bytes(b"broken")
    assert inspector.main(dataset) == 2
    assert "cannot read parquet" in capsys.readouterr().out


def test_metadata_frame_count_mismatch(dataset, capsys):
    path = dataset / "meta/info.json"
    info = json.loads(path.read_text())
    info["total_frames"] += 1
    path.write_text(json.dumps(info))
    assert inspector.main(dataset) == 2
    assert "actual rows 6 != meta.total_frames 7" in capsys.readouterr().out


def test_missing_camera_video(dataset, capsys):
    path = next((dataset / "videos").rglob("*.mp4"))
    path.unlink()
    assert inspector.main(dataset) == 2
    assert "no MP4 files" in capsys.readouterr().out


def test_decode_option_rejects_missing_ffmpeg(dataset, monkeypatch, capsys):
    monkeypatch.setattr(inspector.shutil, "which", lambda name: None)
    assert inspector.main(dataset, check_video_decode=True) == 2
    assert "requires ffmpeg" in capsys.readouterr().out


@pytest.mark.parametrize("returncode", [0, 1])
def test_video_decode_results(dataset, monkeypatch, capsys, returncode):
    calls = []
    monkeypatch.setattr(inspector.shutil, "which", lambda name: "/mock/ffmpeg")

    def decode(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=returncode, stderr="invalid video" if returncode else "")

    monkeypatch.setattr(inspector.subprocess, "run", decode)
    assert inspector.main(dataset, check_video_decode=True) == (2 if returncode else 0)
    assert len(calls) == len(CAMERAS)
    assert all("0:v:0" in argv and "-xerror" in argv for argv in calls)
    output = capsys.readouterr().out
    assert ("decoding failed" if returncode else "full video decoding passed") in output


def test_invalid_timestamp_value(dataset, capsys):
    change_second_file(dataset, lambda df: df.__setitem__("timestamp", [np.nan, np.inf]))
    assert inspector.main(dataset) == 2
    assert "timestamp: 2 invalid values" in capsys.readouterr().out


def test_empty_dataset(tmp_path, capsys):
    assert inspector.check_parquets(tmp_path, {}) == 1
    assert "no parquet files" in capsys.readouterr().out


def test_invalid_metadata(tmp_path, capsys):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{")
    assert inspector.main(tmp_path) == 1
    assert "cannot read metadata" in capsys.readouterr().out
