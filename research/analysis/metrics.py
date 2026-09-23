"""Metrics for comparing fine-tuning methods on held-out task success.

Everything here is deliberately expressed in *absolute* success rate, never as
a ratio against a baseline. H0's held-out (and any zero-shot) success rate can
be near zero, so a ratio would divide by ~0 and manufacture huge, meaningless
"improvements". Comparisons are differences of rates, not quotients.

Pure analysis: it takes plain numbers, no GPU and no robot, so it is testable
on synthetic input (see AGENTS.md).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional

# A mapping from task name to its success rate in [0, 1].
TaskSuccessRates = dict[str, float]


class MetricsInputError(ValueError):
    """Raised on empty input, missing tasks, or out-of-range/invalid data.

    These are caller mistakes we refuse to average over silently: a missing
    task or a bogus rate would quietly bias every downstream comparison.
    """


def _validate_rate(value: object, label: str) -> float:
    """Return `value` as a float, or raise if it is not a rate in [0, 1]."""
    # bool is an int subclass; a stray True/False is almost certainly a bug.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricsInputError(f"{label}: success rate must be a number, got {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise MetricsInputError(f"{label}: success rate is not finite: {value!r}")
    if not (0.0 <= value <= 1.0):
        raise MetricsInputError(f"{label}: success rate must be in [0, 1], got {value}")
    return float(value)


def _select(rates: TaskSuccessRates, tasks: Optional[Iterable[str]]) -> list[float]:
    """Validate and return the success rates for `tasks` (or all of `rates`)."""
    if not rates:
        raise MetricsInputError("rates is empty")
    if tasks is None:
        selected = list(rates.keys())
    else:
        selected = list(tasks)
        if not selected:
            raise MetricsInputError("tasks is empty")
        missing = [t for t in selected if t not in rates]
        if missing:
            raise MetricsInputError(f"rates is missing tasks: {sorted(missing)}")
    return [_validate_rate(rates[t], t) for t in selected]


def mean_success_rate(
    rates: TaskSuccessRates, tasks: Optional[Iterable[str]] = None
) -> float:
    """Unweighted mean success rate over `tasks` (default: every task in `rates`)."""
    return statistics.fmean(_select(rates, tasks))


def cross_task_variance(
    rates: TaskSuccessRates, tasks: Optional[Iterable[str]] = None
) -> float:
    """Population standard deviation of success rate across tasks.

    A measure of how unevenly a method does across the held-out suite: a high
    mean hides nothing if the spread is also high. Requires at least 2 tasks.
    """
    values = _select(rates, tasks)
    if len(values) < 2:
        raise MetricsInputError("cross_task_variance requires at least 2 tasks")
    return statistics.pstdev(values)


@dataclass(frozen=True)
class MethodComparison:
    """One task's success rate under each method being compared."""

    task: str
    rates: dict[str, float]  # method name -> success rate on this task


def compare_methods(
    results: dict[str, TaskSuccessRates], tasks: Optional[Iterable[str]] = None
) -> list[MethodComparison]:
    """Reshape per-method task results into a per-task view.

    `results` maps a method name to its per-task success rates. The output is
    one row per task, each carrying that task's rate under every method, so
    methods can be compared task by task. `tasks` defaults to the tasks common
    to all methods; pass it explicitly to require a specific set (every method
    must then cover all of them).
    """
    if not results:
        raise MetricsInputError("results is empty")
    method_names = list(results.keys())
    for m in method_names:
        if not results[m]:
            raise MetricsInputError(f"method '{m}' has no task results")

    if tasks is None:
        common = set(results[method_names[0]])
        for m in method_names[1:]:
            common &= set(results[m])
        if not common:
            raise MetricsInputError("no tasks are common to all methods")
        selected = sorted(common)
    else:
        selected = list(tasks)
        if not selected:
            raise MetricsInputError("tasks is empty")
        for m in method_names:
            missing = [t for t in selected if t not in results[m]]
            if missing:
                raise MetricsInputError(f"method '{m}' is missing tasks: {sorted(missing)}")

    comparisons = []
    for task in selected:
        row = {m: _validate_rate(results[m][task], f"{m}/{task}") for m in method_names}
        comparisons.append(MethodComparison(task=task, rates=row))
    return comparisons


@dataclass(frozen=True)
class DataEfficiencyPoint:
    """Success rate reached after fine-tuning on `num_demos` demonstrations."""

    num_demos: int
    success_rate: float


def data_efficiency_curve(
    points: Iterable[DataEfficiencyPoint],
) -> list[DataEfficiencyPoint]:
    """Validate points and return them sorted by ascending demo count."""
    points = list(points)
    if not points:
        raise MetricsInputError("points is empty")
    for p in points:
        if isinstance(p.num_demos, bool) or not isinstance(p.num_demos, int):
            raise MetricsInputError(f"num_demos must be an int, got {p.num_demos!r}")
        if p.num_demos < 0:
            raise MetricsInputError(f"num_demos must be >= 0, got {p.num_demos}")
        _validate_rate(p.success_rate, f"num_demos={p.num_demos}")
    return sorted(points, key=lambda p: p.num_demos)


def demos_to_reach(
    curve: Iterable[DataEfficiencyPoint], target_success_rate: float
) -> Optional[int]:
    """Fewest demos at which success rate first reaches `target_success_rate`.

    Returns None if no point on the curve hits the target. Success rate need
    not be monotonic in demo count, so we scan by ascending demos and return
    the first crossing.
    """
    curve = list(curve)
    if not curve:
        raise MetricsInputError("curve is empty")
    _validate_rate(target_success_rate, "target_success_rate")
    for p in sorted(curve, key=lambda p: p.num_demos):
        if p.success_rate >= target_success_rate:
            return p.num_demos
    return None
