# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy
from pathlib import Path

import pytest
from experiments.streak_dependence_audit import (
    DEFAULT_SEEDS,
    PRIVATE_TERMS,
    CensoredStreakTracker,
    RoundPredictorRow,
    RunRecord,
    StreakDependenceAuditConfig,
    autocorrelation,
    continuation_diagnostics,
    evaluate_model,
    fit_ols,
    geometric_pmf,
    geometric_shape_diagnostics,
    geometric_survival,
    model_features,
    predictive_model_rows,
    run_streak_dependence_audit,
    score_band,
    strategy_relevance_gate,
    streak_group,
    transition_rows,
)


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (("win", "win", "push", "win"), [("win", 3)]),
        (("loss", "loss", "push", "loss"), [("loss", 3)]),
        (("win", "push", "loss"), [("win", 1), ("loss", 1)]),
    ],
)
def test_push_neutral_streak_semantics(outcomes, expected) -> None:
    tracker = CensoredStreakTracker()
    tracker.start_measurement()
    for outcome in outcomes:
        tracker.observe(outcome)
    tracker.finalize_measurement()
    assert [(record.kind, record.length) for record in tracker.records] == expected


def test_split_net_zero_push_leaves_live_streak_unchanged() -> None:
    tracker = CensoredStreakTracker()
    tracker.observe_net(10.0)
    before = (tracker.sign, tracker.current_length)
    tracker.observe_net(10.0 + -10.0)
    assert (tracker.sign, tracker.current_length) == before == (1, 1)
    tracker.observe_net(-10.0)
    assert (tracker.sign, tracker.current_length) == (-1, 1)


def test_burn_in_state_and_boundary_censoring_are_preserved() -> None:
    tracker = CensoredStreakTracker()
    tracker.observe("win")
    tracker.observe("win")
    tracker.start_measurement()
    assert (tracker.sign, tracker.current_length) == (1, 2)
    tracker.observe("push")
    tracker.observe("win")
    tracker.observe("loss")
    tracker.observe("win")
    tracker.finalize_measurement()
    assert tracker.records == [
        RunRecord("win", 3, left_censored=True),
        RunRecord("loss", 1),
        RunRecord("win", 1, right_censored=True),
    ]
    primary = [
        record
        for record in tracker.records
        if not record.left_censored and not record.right_censored
    ]
    assert primary == [RunRecord("loss", 1)]


def test_seed_trackers_never_join_runs() -> None:
    first = CensoredStreakTracker()
    second = CensoredStreakTracker()
    for tracker in (first, second):
        tracker.start_measurement()
        tracker.observe("win")
        tracker.finalize_measurement()
    assert [record.length for record in (*first.records, *second.records)] == [1, 1]


def test_geometric_benchmark_hand_calculations() -> None:
    assert geometric_pmf(0.6, 1) == pytest.approx(0.4)
    assert geometric_pmf(0.6, 3) == pytest.approx(0.144)
    assert geometric_survival(0.6, 3) == pytest.approx(0.36)
    assert geometric_survival(0.6, 21) == pytest.approx(0.6**20)
    assert 1 / (1 - 0.6) == pytest.approx(2.5)


def test_shape_diagnostics_exact_match_and_overflow() -> None:
    result = geometric_shape_diagnostics(
        {1: 2, 2: 1, 3: 1},
        continuation=0.5,
        max_exact=2,
        tail_thresholds=(2, 3),
    )
    assert result["observed"] == {1: 2, 2: 1, 3: 1}
    assert result["total_variation_distance"] == pytest.approx(0.0)
    assert result["chi_square_descriptive"] == pytest.approx(0.0)
    assert result["maximum_survival_deviation"] == pytest.approx(0.0)
    assert result["observed_expected_ratio"][3] == pytest.approx(1.0)
    assert result["tails"]["3"]["ratio"] == pytest.approx(1.0)


def test_continuation_uses_only_observable_risk() -> None:
    records = [
        RunRecord("win", 3, left_censored=True),
        RunRecord("win", 1),
        RunRecord("win", 2, right_censored=True),
    ]
    rows = continuation_diagnostics(records, (1, 2), expected=0.5)
    assert rows[0]["continued"] == 2
    assert rows[0]["number_at_risk"] == 3
    assert rows[0]["empirical_continuation"] == pytest.approx(2 / 3)
    assert rows[1]["continued"] == 1
    assert rows[1]["number_at_risk"] == 1


def test_transition_matrices_compress_and_retain_pushes() -> None:
    raw = ["W", "P", "W", "L"]
    resolved = [value for value in raw if value != "P"]
    resolved_rows = transition_rows(resolved, ("W", "L"))
    assert (
        next(
            row for row in resolved_rows if row["current"] == "W" and row["next"] == "W"
        )["count"]
        == 1
    )
    raw_rows = transition_rows(raw, ("W", "P", "L"))
    assert (
        next(row for row in raw_rows if row["current"] == "W" and row["next"] == "P")[
            "count"
        ]
        == 1
    )


def test_autocorrelation_at_configured_lag() -> None:
    assert autocorrelation([1.0, -1.0, 1.0, -1.0], 1) == pytest.approx(-0.75)
    assert autocorrelation([1.0, -1.0, 1.0, -1.0], 4) is None


def test_streak_groups_and_fixed_score_bands() -> None:
    assert streak_group(0, 0) == "none"
    assert streak_group(1, 4) == "win_4"
    assert streak_group(-1, 7) == "loss_5_plus"
    assert score_band(-0.0026) == "high-rich"
    assert score_band(-0.0025) == "neutral"
    assert score_band(0.0025) == "neutral"
    assert score_band(0.0026) == "low-rich"


def synthetic_rows(count: int = 40) -> list[RoundPredictorRow]:
    rows = []
    for index in range(count):
        score = (index % 7 - 3) / 100.0
        sign = (-1, 0, 1)[index % 3]
        length = index % 5 if sign else 0
        net = 0.2 + 2 * score + 0.1 * sign * length - 0.03 * length + score * sign
        outcome = "win" if net > 0 else "loss"
        rows.append(RoundPredictorRow(score, sign, length, net, outcome, 0, 0))
    return rows


def test_model_designs_interaction_and_ols() -> None:
    row = RoundPredictorRow(0.2, -1, 3, -1.0, "loss", 0, 0)
    assert model_features(row, "A") == (1.0, 0.2)
    assert model_features(row, "B") == (1.0, 0.2, -3.0, 3.0)
    assert model_features(row, "C") == (1.0, 0.2, -3.0, 3.0, -0.2)
    rows = synthetic_rows()
    coefficients = fit_ols(rows[:20], "C")
    performance = evaluate_model(rows[20:], "C", coefficients)
    assert performance["mse"] == pytest.approx(0.0, abs=1e-20)


def test_predictive_split_is_contiguous_and_improvement_is_positive() -> None:
    rows = synthetic_rows()
    coefficients, performance = predictive_model_rows(rows, "one2six", 72)
    assert all(row["estimation_rows"] == 20 for row in coefficients)
    monetary = {row["model"]: row for row in performance if row["target"] == "monetary"}
    assert monetary["A"]["evaluation_rows"] == 20
    assert monetary["A"]["mse"] - monetary["C"]["mse"] > 0


def summary(mean=0.01, interval=(0.001, 0.02), positive=8):
    return {
        "mean": mean,
        "student_t_95_ci": list(interval),
        "positive_differences": positive,
    }


def test_strategy_relevance_gate_passes_and_each_condition_can_fail() -> None:
    one = summary()
    paired = summary()
    iid = summary(mean=0.0, interval=(-0.01, 0.01), positive=5)
    assert strategy_relevance_gate(one, paired, iid)
    failures = [
        (summary(mean=-0.01), paired, iid),
        (summary(interval=(-0.01, 0.02)), paired, iid),
        (summary(positive=7), paired, iid),
        (one, summary(interval=(-0.01, 0.02)), iid),
        (one, paired, summary()),
    ]
    assert all(not strategy_relevance_gate(*case) for case in failures)


def test_config_defaults_and_validation() -> None:
    assert DEFAULT_SEEDS == tuple(range(72, 82))
    with pytest.raises(ValueError, match="unique"):
        StreakDependenceAuditConfig(seeds=(72, 72))
    with pytest.raises(ValueError, match="frozen"):
        StreakDependenceAuditConfig(current_rack_weight=0.9)


def test_small_end_to_end_export_is_private(tmp_path: Path) -> None:
    output = tmp_path / "streak-audit"
    summary_payload = run_streak_dependence_audit(
        StreakDependenceAuditConfig(
            seeds=(72,),
            rounds_per_seed=200,
            burn_in_rounds=20,
            output_dir=output,
        )
    )
    assert summary_payload["hidden_state_exported"] is False
    assert summary_payload["aggregate"]["physical_iid"]["rounds"] == 200
    required = {
        "summary.json",
        "summary.md",
        "streak_distribution_full.csv",
        "per_seed_predictive_models.csv",
        "conditional_next_round_by_streak_and_score.csv",
    }
    assert required.issubset(path.name for path in output.iterdir())
    exported = "".join(
        path.read_text(encoding="utf-8")
        for path in output.iterdir()
        if path.suffix in {".json", ".csv", ".md"}
    ).lower()
    assert not any(term in exported for term in PRIVATE_TERMS)
    frozen = deepcopy(summary_payload["config"]["frozen_weights"])
    assert frozen == summary_payload["config"]["frozen_weights"]
