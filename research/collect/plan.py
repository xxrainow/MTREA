"""Collection plans: which concrete instruction is recorded in which session.

lerobot-record attaches one instruction (--dataset.single_task) to a whole run,
so a T_sub family is collected as several sessions, one per object, appended to
one dataset. The plan yaml fixes those sessions and the per-episode scene
layout before recording, so the result can be checked against it afterwards.

Pure logic: no hardware, no lerobot, no lab/ or robot/ imports.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).resolve().parent / "tasks"

_PLAN_KEYS = {
    "task", "dataset", "seed", "target", "distractor", "positions",
    "objects", "sessions", "episode_time_s", "reset_time_s",
}
_SESSION_KEYS = {"color", "type", "episodes"}
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")

# Distractor kinds. Both differ from the target object in color; a same-type
# distractor makes the color word necessary, an other-type one the type word.
SAME_TYPE = "same_type"
OTHER_TYPE = "other_type"


class PlanError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class ObjectSpec:
    color: str
    type: str

    def __str__(self) -> str:
        return f"{self.color} {self.type}"


@dataclass(frozen=True)
class EpisodeLayout:
    target_position: str
    distractor: ObjectSpec | None
    distractor_position: str | None
    distractor_kind: str | None = None  # SAME_TYPE or OTHER_TYPE


@dataclass(frozen=True)
class Session:
    index: int  # 1-based, as the operator sees it
    object: ObjectSpec
    instruction: str
    num_episodes: int
    schedule: tuple[EpisodeLayout, ...]


@dataclass(frozen=True)
class Plan:
    path: Path
    task: str
    instruction_template: str
    dataset: str
    seed: int
    target: str
    distractor: bool
    positions: tuple[str, ...]
    objects: tuple[ObjectSpec, ...]  # inventory available as distractors
    session_objects: tuple[tuple[ObjectSpec, int], ...]
    episode_time_s: float
    reset_time_s: float

    @property
    def repo_id(self) -> str:
        return f"local/{self.dataset}"

    @property
    def raw_subpath(self) -> Path:
        """Dataset root relative to the data root (data/raw/<dataset>)."""
        return Path("raw") / self.dataset

    @property
    def total_episodes(self) -> int:
        return sum(n for _, n in self.session_objects)


def render_instruction(template: str, obj: ObjectSpec, target: str) -> str:
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    unknown = fields - {"object", "target"}
    if unknown:
        raise PlanError(f"instruction template has unknown fields {sorted(unknown)}: {template!r}")
    text = template.format(object=str(obj), target=target)
    if "{" in text or "}" in text:
        raise PlanError(f"instruction has leftover braces: {text!r}")
    return text


def _require(raw: dict, key: str, kind: type | tuple[type, ...]):
    if key not in raw:
        raise PlanError(f"missing key {key!r}")
    value = raw[key]
    # bool is an int subclass; a seed of `true` must not pass as 1.
    if isinstance(value, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,)):
        raise PlanError(f"{key!r} must be {kind}, got {value!r}")
    if not isinstance(value, kind):
        raise PlanError(f"{key!r} must be {kind}, got {value!r}")
    return value


def _names(raw: dict, key: str) -> tuple[str, ...]:
    values = _require(raw, key, list)
    if not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise PlanError(f"{key!r} must be a non-empty list of names")
    if len(set(values)) != len(values):
        raise PlanError(f"{key!r} has duplicates: {values}")
    return tuple(values)


def parse_plan(raw: dict, path: Path, task_spec: dict) -> Plan:
    """Build a Plan from already-loaded yaml and validate it against the task."""
    if not isinstance(raw, dict):
        raise PlanError("plan must be a mapping")
    unknown = set(raw) - _PLAN_KEYS
    if unknown:
        raise PlanError(f"unknown plan keys {sorted(unknown)}")

    task = _require(raw, "task", str)
    if task_spec.get("name") != task:
        raise PlanError(f"plan task {task!r} != task file name {task_spec.get('name')!r}")
    # Held-out tasks are never teleop-recorded for training (protocol.md).
    if task_spec.get("role") != "t_sub":
        raise PlanError(f"task {task!r} has role {task_spec.get('role')!r}; only t_sub tasks are collected")
    template = task_spec.get("instruction_template")
    if not isinstance(template, str):
        raise PlanError(f"task {task!r} has no instruction_template")
    target_episodes = task_spec.get("target_episodes")
    if not isinstance(target_episodes, int) or isinstance(target_episodes, bool):
        raise PlanError(f"task {task!r} has no integer target_episodes")

    dataset = _require(raw, "dataset", str)
    if not _NAME.match(dataset):
        raise PlanError(f"dataset name {dataset!r} must be a plain directory name")
    seed = _require(raw, "seed", int)
    target = _require(raw, "target", str)
    distractor = _require(raw, "distractor", bool)
    episode_time_s = _require(raw, "episode_time_s", (int, float))
    reset_time_s = _require(raw, "reset_time_s", (int, float))
    if episode_time_s <= 0 or reset_time_s < 0:
        raise PlanError("episode_time_s must be > 0 and reset_time_s >= 0")

    positions = _names(raw, "positions")
    if distractor and len(positions) < 2:
        raise PlanError("a distractor needs at least 2 positions")

    objects_raw = _require(raw, "objects", dict)
    if set(objects_raw) != {"types", "colors"}:
        raise PlanError("'objects' must have exactly 'types' and 'colors'")
    types = _names(objects_raw, "types")
    colors = _names(objects_raw, "colors")
    inventory = tuple(ObjectSpec(c, t) for t in types for c in colors)

    sessions_raw = _require(raw, "sessions", list)
    if not sessions_raw:
        raise PlanError("'sessions' is empty")
    session_objects = []
    for i, s in enumerate(sessions_raw, start=1):
        if not isinstance(s, dict) or set(s) != _SESSION_KEYS:
            raise PlanError(f"session {i} must have exactly {sorted(_SESSION_KEYS)}")
        obj = ObjectSpec(_require(s, "color", str), _require(s, "type", str))
        episodes = _require(s, "episodes", int)
        if obj not in inventory:
            raise PlanError(f"session {i}: {obj} is not in objects (types x colors)")
        # Fewer episodes than positions would leave a position uncovered.
        if episodes < len(positions):
            raise PlanError(f"session {i}: {episodes} episodes < {len(positions)} positions")
        session_objects.append((obj, episodes))

    plan = Plan(
        path=path, task=task, instruction_template=template, dataset=dataset, seed=seed,
        target=target, distractor=distractor, positions=positions, objects=inventory,
        session_objects=tuple(session_objects),
        episode_time_s=episode_time_s, reset_time_s=reset_time_s,
    )
    _check_protocol(plan, target_episodes)
    for obj, _ in plan.session_objects:
        render_instruction(template, obj, target)
    if distractor:
        for obj, _ in plan.session_objects:
            for kind, candidates in distractor_candidates(plan, obj).items():
                if not candidates:
                    raise PlanError(f"objects have no {kind} distractor of another color for {obj}")
    return plan


def _check_protocol(plan: Plan, target_episodes: int) -> None:
    """Object diversity rules from protocol.md, counted over planned episodes."""
    total = plan.total_episodes
    if total != target_episodes:
        raise PlanError(f"plan has {total} episodes, task target_episodes is {target_episodes}")
    objs = [obj for obj, _ in plan.session_objects]
    if len(set(objs)) != len(objs):
        raise PlanError("one session per object: duplicate (color, type) in sessions")
    colors = {o.color for o in objs}
    types = {o.type for o in objs}
    if len(colors) < 3:
        raise PlanError(f"protocol needs >= 3 colors, plan has {sorted(colors)}")
    if len(types) < 2:
        raise PlanError(f"protocol needs >= 2 object types, plan has {sorted(types)}")
    per_combo = Counter()
    for obj, n in plan.session_objects:
        per_combo[obj] += n
    for obj, n in per_combo.items():
        if 2 * n > target_episodes:
            raise PlanError(f"{obj} has {n} of {target_episodes} episodes (> 50%)")


def load_plan(path: str | Path, tasks_dir: Path = TASKS_DIR) -> Plan:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("task"), str):
        raise PlanError(f"{path}: plan must be a mapping with a 'task' name")
    task_path = tasks_dir / f"{raw['task']}.yaml"
    if not task_path.exists():
        raise PlanError(f"{path}: task file {task_path} not found")
    task_spec = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    try:
        return parse_plan(raw, path, task_spec)
    except PlanError as exc:
        raise PlanError(f"{path}: {exc}") from None


def distractor_candidates(plan: Plan, obj: ObjectSpec) -> dict[str, list[ObjectSpec]]:
    """Objects of another color, by kind: same type as obj, or another type."""
    other_color = [o for o in plan.objects if o.color != obj.color]
    return {
        SAME_TYPE: sorted(o for o in other_color if o.type == obj.type),
        OTHER_TYPE: sorted(o for o in other_color if o.type != obj.type),
    }


def _balanced(rng: random.Random, items: list[ObjectSpec], count: int) -> list[ObjectSpec]:
    """count picks, every item once per round, round order shuffled."""
    picks: list[ObjectSpec] = []
    while len(picks) < count:
        round_ = list(items)
        rng.shuffle(round_)
        picks.extend(round_)
    return picks[:count]


def _schedule(plan: Plan, index: int, obj: ObjectSpec, n: int) -> tuple[EpisodeLayout, ...]:
    # A string seed is hashed deterministically (unlike hash()), across platforms.
    rng = random.Random(f"{plan.seed}/{index}")
    order = list(plan.positions)
    rng.shuffle(order)
    targets = [order[i % len(order)] for i in range(n)]
    if not plan.distractor:
        return tuple(EpisodeLayout(t, None, None) for t in targets)

    # Kinds alternate, so each session needs both the color and the type word;
    # the seed picks which kind comes first (and gets the extra one when n is odd).
    first, second = (SAME_TYPE, OTHER_TYPE) if rng.random() < 0.5 else (OTHER_TYPE, SAME_TYPE)
    kinds = [first if i % 2 == 0 else second for i in range(n)]
    candidates = distractor_candidates(plan, obj)
    pools = {kind: _balanced(rng, candidates[kind], kinds.count(kind)) for kind in (first, second)}
    distractors = [pools[kind].pop(0) for kind in kinds]

    # Distractor positions are balanced like target positions, so neither
    # becomes a spatial shortcut: each round of len(positions) episodes uses
    # every position at most once, drawn until none collides with its target.
    d_positions: list[str] = []
    width = len(plan.positions)
    for start in range(0, n, width):
        round_targets = targets[start:start + width]
        while True:
            draw = rng.sample(plan.positions, len(round_targets))
            if all(d != t for d, t in zip(draw, round_targets)):
                break
        d_positions.extend(draw)
    return tuple(EpisodeLayout(t, d, p, k)
                 for t, d, p, k in zip(targets, distractors, d_positions, kinds))


def sessions(plan: Plan) -> list[Session]:
    result = []
    for index, (obj, n) in enumerate(plan.session_objects, start=1):
        result.append(Session(
            index=index,
            object=obj,
            instruction=render_instruction(plan.instruction_template, obj, plan.target),
            num_episodes=n,
            schedule=_schedule(plan, index, obj, n),
        ))
    return result
