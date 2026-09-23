"""Tests for research.analysis.metrics — synthetic numbers only, no GPU/robot."""

import pytest

from research.analysis.metrics import (
    DataEfficiencyPoint,
    MethodComparison,
    MetricsInputError,
    compare_methods,
    cross_task_variance,
    data_efficiency_curve,
    demos_to_reach,
    mean_success_rate,
)


# --- mean_success_rate ---

def test_mean_success_rate_all_tasks():
    rates = {"a": 0.0, "b": 0.5, "c": 1.0}
    assert mean_success_rate(rates) == pytest.approx(0.5)


def test_mean_success_rate_subset():
    rates = {"a": 0.2, "b": 0.8, "c": 1.0}
    assert mean_success_rate(rates, tasks=["a", "b"]) == pytest.approx(0.5)


def test_mean_success_rate_empty_raises():
    with pytest.raises(MetricsInputError):
        mean_success_rate({})


def test_mean_success_rate_missing_task_raises():
    with pytest.raises(MetricsInputError):
        mean_success_rate({"a": 0.5}, tasks=["a", "b"])


def test_mean_success_rate_empty_tasks_raises():
    with pytest.raises(MetricsInputError):
        mean_success_rate({"a": 0.5}, tasks=[])


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), "0.5", True])
def test_mean_success_rate_invalid_rate_raises(bad):
    with pytest.raises(MetricsInputError):
        mean_success_rate({"a": bad})


# --- cross_task_variance (population stdev) ---

def test_cross_task_variance_value():
    # population stdev of {0.0, 1.0} is 0.5
    assert cross_task_variance({"a": 0.0, "b": 1.0}) == pytest.approx(0.5)


def test_cross_task_variance_identical_is_zero():
    assert cross_task_variance({"a": 0.7, "b": 0.7, "c": 0.7}) == pytest.approx(0.0)


def test_cross_task_variance_requires_two_tasks():
    with pytest.raises(MetricsInputError):
        cross_task_variance({"a": 0.5})


def test_cross_task_variance_empty_raises():
    with pytest.raises(MetricsInputError):
        cross_task_variance({})


# --- compare_methods ---

def test_compare_methods_common_tasks_default():
    results = {
        "H0": {"pick": 0.1, "stack": 0.0, "extra": 0.9},
        "H1": {"pick": 0.6, "stack": 0.4},
    }
    out = compare_methods(results)
    assert out == [
        MethodComparison(task="pick", rates={"H0": 0.1, "H1": 0.6}),
        MethodComparison(task="stack", rates={"H0": 0.0, "H1": 0.4}),
    ]


def test_compare_methods_explicit_tasks_preserves_order():
    results = {
        "H0": {"pick": 0.1, "stack": 0.0},
        "H1": {"pick": 0.6, "stack": 0.4},
    }
    out = compare_methods(results, tasks=["stack", "pick"])
    assert [c.task for c in out] == ["stack", "pick"]


def test_compare_methods_empty_raises():
    with pytest.raises(MetricsInputError):
        compare_methods({})


def test_compare_methods_method_without_tasks_raises():
    with pytest.raises(MetricsInputError):
        compare_methods({"H0": {}})


def test_compare_methods_no_common_tasks_raises():
    with pytest.raises(MetricsInputError):
        compare_methods({"H0": {"a": 0.1}, "H1": {"b": 0.2}})


def test_compare_methods_explicit_missing_task_raises():
    results = {"H0": {"pick": 0.1}, "H1": {"pick": 0.6}}
    with pytest.raises(MetricsInputError):
        compare_methods(results, tasks=["pick", "stack"])


def test_compare_methods_invalid_rate_raises():
    with pytest.raises(MetricsInputError):
        compare_methods({"H0": {"pick": 1.5}, "H1": {"pick": 0.6}})


# --- data_efficiency_curve ---

def test_data_efficiency_curve_sorts_by_demos():
    points = [
        DataEfficiencyPoint(num_demos=50, success_rate=0.6),
        DataEfficiencyPoint(num_demos=10, success_rate=0.2),
        DataEfficiencyPoint(num_demos=30, success_rate=0.5),
    ]
    curve = data_efficiency_curve(points)
    assert [p.num_demos for p in curve] == [10, 30, 50]


def test_data_efficiency_curve_empty_raises():
    with pytest.raises(MetricsInputError):
        data_efficiency_curve([])


def test_data_efficiency_curve_negative_demos_raises():
    with pytest.raises(MetricsInputError):
        data_efficiency_curve([DataEfficiencyPoint(num_demos=-1, success_rate=0.5)])


def test_data_efficiency_curve_non_int_demos_raises():
    with pytest.raises(MetricsInputError):
        data_efficiency_curve([DataEfficiencyPoint(num_demos=10.0, success_rate=0.5)])


def test_data_efficiency_curve_invalid_rate_raises():
    with pytest.raises(MetricsInputError):
        data_efficiency_curve([DataEfficiencyPoint(num_demos=10, success_rate=2.0)])


# --- demos_to_reach ---

def test_demos_to_reach_returns_first_crossing():
    curve = data_efficiency_curve([
        DataEfficiencyPoint(num_demos=10, success_rate=0.2),
        DataEfficiencyPoint(num_demos=30, success_rate=0.55),
        DataEfficiencyPoint(num_demos=50, success_rate=0.9),
    ])
    assert demos_to_reach(curve, 0.5) == 30


def test_demos_to_reach_target_never_met_returns_none():
    curve = [
        DataEfficiencyPoint(num_demos=10, success_rate=0.2),
        DataEfficiencyPoint(num_demos=50, success_rate=0.4),
    ]
    assert demos_to_reach(curve, 0.9) is None


def test_demos_to_reach_non_monotonic_uses_fewest_demos():
    curve = [
        DataEfficiencyPoint(num_demos=40, success_rate=0.4),
        DataEfficiencyPoint(num_demos=20, success_rate=0.8),  # fewer demos, higher rate
    ]
    assert demos_to_reach(curve, 0.7) == 20


def test_demos_to_reach_empty_raises():
    with pytest.raises(MetricsInputError):
        demos_to_reach([], 0.5)


def test_demos_to_reach_invalid_target_raises():
    curve = [DataEfficiencyPoint(num_demos=10, success_rate=0.5)]
    with pytest.raises(MetricsInputError):
        demos_to_reach(curve, 1.5)
