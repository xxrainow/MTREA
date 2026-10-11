"""Tests for research.collect.plan — pure logic, no hardware."""

import copy
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from research.collect.plan import (
    TASKS_DIR, ObjectSpec, PlanError, distractor_candidates, load_plan, parse_plan, sessions,
)

PLAN_PATH = Path(__file__).resolve().parent / "plans" / "pick_place_v1.yaml"
TASK = yaml.safe_load((TASKS_DIR / "pick_place.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def raw():
    return yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))


def parse(raw, task=TASK):
    return parse_plan(raw, PLAN_PATH, task)


def test_committed_plan_loads():
    plan = load_plan(PLAN_PATH)
    assert plan.total_episodes == TASK["target_episodes"] == 50
    assert plan.repo_id == "local/pick_place_v1"
    assert plan.raw_subpath == Path("raw", "pick_place_v1")
    assert [s.num_episodes for s in sessions(plan)] == [9, 8, 8, 9, 8, 8]
    assert [s.index for s in sessions(plan)] == [1, 2, 3, 4, 5, 6]


def test_rejects_wrong_total(raw):
    raw["sessions"][0]["episodes"] += 1
    with pytest.raises(PlanError, match="51 episodes"):
        parse(raw)


def test_rejects_fewer_than_three_colors(raw):
    raw["sessions"] = [
        {"color": "red", "type": "block", "episodes": 13},
        {"color": "blue", "type": "block", "episodes": 12},
        {"color": "red", "type": "cup", "episodes": 13},
        {"color": "blue", "type": "cup", "episodes": 12},
    ]
    with pytest.raises(PlanError, match=">= 3 colors"):
        parse(raw)


def test_rejects_fewer_than_two_types(raw):
    raw["sessions"] = [
        {"color": "red", "type": "block", "episodes": 17},
        {"color": "blue", "type": "block", "episodes": 17},
        {"color": "green", "type": "block", "episodes": 16},
    ]
    with pytest.raises(PlanError, match=">= 2 object types"):
        parse(raw)


def test_rejects_combo_over_half(raw):
    raw["sessions"] = [
        {"color": "red", "type": "block", "episodes": 26},
        {"color": "blue", "type": "cup", "episodes": 12},
        {"color": "green", "type": "cup", "episodes": 12},
    ]
    with pytest.raises(PlanError, match="> 50%"):
        parse(raw)


def test_accepts_combo_at_exactly_half(raw):
    raw["sessions"] = [
        {"color": "red", "type": "block", "episodes": 25},
        {"color": "blue", "type": "cup", "episodes": 13},
        {"color": "green", "type": "cup", "episodes": 12},
    ]
    parse(raw)


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.update(distractor="yes"), "distractor"),
    (lambda r: r.update(seed=True), "seed"),
    (lambda r: r.pop("seed"), "missing key 'seed'"),
    (lambda r: r.update(extra=1), "unknown plan keys"),
    (lambda r: r["sessions"][0].update(color="purple"), "not in objects"),
    (lambda r: r["sessions"].append(dict(r["sessions"][0], episodes=0)), "positions"),
])
def test_rejects_malformed(raw, mutate, match):
    mutate(raw)
    with pytest.raises(PlanError, match=match):
        parse(raw)


def test_rejects_duplicate_session_object(raw):
    raw["sessions"][1] = dict(raw["sessions"][0], episodes=8)
    with pytest.raises(PlanError, match="duplicate"):
        parse(raw)


def test_rejects_held_out_task(raw):
    with pytest.raises(PlanError, match="t_sub"):
        parse(raw, dict(TASK, role="held_out"))


def test_rejects_unknown_template_field(raw):
    task = dict(TASK, instruction_template="pick up the {object} with {gripper}")
    with pytest.raises(PlanError, match="unknown fields"):
        parse(raw, task)


def test_rejects_leftover_braces(raw):
    task = dict(TASK, instruction_template="pick up the {object} {{and}} place it in the {target}")
    with pytest.raises(PlanError, match="leftover braces"):
        parse(raw, task)


def test_instructions_render_from_template():
    plan = load_plan(PLAN_PATH)
    template = TASK["instruction_template"]
    for s in sessions(plan):
        assert s.instruction == template.format(object=str(s.object), target="bowl")
        assert "{" not in s.instruction and "}" not in s.instruction
    assert sessions(plan)[0].instruction == "pick up the red block and place it in the bowl"


def test_schedule_is_deterministic_for_a_seed(raw):
    a = sessions(parse(raw))
    b = sessions(parse(copy.deepcopy(raw)))
    assert a == b
    raw["seed"] += 1
    c = sessions(parse(raw))
    assert [s.schedule for s in a] != [s.schedule for s in c]


def test_schedule_invariants():
    plan = load_plan(PLAN_PATH)
    for s in sessions(plan):
        assert len(s.schedule) == s.num_episodes
        assert {e.target_position for e in s.schedule} == set(plan.positions)
        for e in s.schedule:
            assert e.distractor is not None
            assert e.distractor != s.object
            assert e.distractor.color != s.object.color
            assert e.distractor.type != s.object.type  # possible with 2 types
            assert e.distractor_position != e.target_position
            assert e.distractor_position in plan.positions
        # Both eligible distractors are used, balanced within one.
        counts = Counter(e.distractor for e in s.schedule)
        assert len(counts) == 2 and max(counts.values()) - min(counts.values()) <= 1
        # Distractor positions are balanced like target positions (no spatial shortcut).
        d_counts = Counter(e.distractor_position for e in s.schedule)
        assert set(d_counts) == set(plan.positions)
        assert max(d_counts.values()) - min(d_counts.values()) <= 1


def test_distractor_falls_back_to_same_type_when_only_one_type(raw):
    blocks_only = replace(parse(raw), objects=tuple(
        ObjectSpec(c, "block") for c in ("red", "blue", "green")))
    assert distractor_candidates(blocks_only, ObjectSpec("red", "block")) == [
        ObjectSpec("blue", "block"), ObjectSpec("green", "block")]


def test_no_distractor_leaves_fields_empty(raw):
    raw["distractor"] = False
    for s in sessions(parse(raw)):
        assert all(e.distractor is None and e.distractor_position is None for e in s.schedule)
        assert {e.target_position for e in s.schedule} == {"P1", "P2", "P3", "P4", "P5"}
