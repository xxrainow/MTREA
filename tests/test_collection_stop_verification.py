import pytest

from lab.core import dataset_metadata


@pytest.mark.parametrize(
    "after,partial,exit_code,has_problem",
    [(6, True, 0, False), (3, True, 0, False),
     (6, False, 0, True), (11, True, 0, True),
     (2, True, 0, True), (6, True, 1, True)],
)
def test_partial_stop_only_relaxes_expected_count(
    tmp_path, monkeypatch, after, partial, exit_code, has_problem,
):
    monkeypatch.setattr(dataset_metadata, "read_episode_count", lambda _: after)
    monkeypatch.setattr(dataset_metadata, "read_tasks", lambda _: ["Move."])
    monkeypatch.setattr(
        dataset_metadata, "read_episode_tasks",
        lambda _: {index: ["Move."] for index in range(3, after)},
    )
    _, problems = dataset_metadata.verify(
        tmp_path, 3, 7, "Move.", exit_code, allow_partial=partial,
    )
    assert bool(problems) is has_problem


def test_partial_stop_still_checks_episode_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_metadata, "read_episode_count", lambda _: 4)
    monkeypatch.setattr(dataset_metadata, "read_tasks", lambda _: ["Move."])
    monkeypatch.setattr(dataset_metadata, "read_episode_tasks", lambda _: {})
    _, problems = dataset_metadata.verify(
        tmp_path, 3, 7, "Move.", 0, allow_partial=True,
    )
    assert problems
