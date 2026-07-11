# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import pytest
from experiments.conditional_profitability import (
    BAND_SCORES,
    DEFAULT_VALIDATION_SEEDS,
    PRIVATE_TERMS,
    SCORE_BANDS,
    BandAccumulator,
    ConditionalProfitabilityConfig,
    DecisionFrequencyRecorder,
    RoundMonetaryObservation,
    ScoreBandCutpoints,
    _aggregate_band_metrics,
    _frozen_state,
    _new_trajectory,
    _ordered_band_trends,
    _paired_monetary_slopes,
    _score_band_contrasts,
    assign_score_band,
    decision_hand_category,
    deterministic_permutation_regression,
    evaluate_candidate_state,
    freeze_score_cutpoints,
    linear_quantile,
    run_conditional_profitability_experiment,
)
from experiments.fading_exclusion_validation import RegressionDiagnostics
from experiments.observable_card_response import CardIndicators

from shufflemaster_sim.actions import ActionType
from shufflemaster_sim.state import BlackjackDecisionState
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)


def test_default_validation_seeds_are_exactly_52_through_61() -> None:
    assert DEFAULT_VALIDATION_SEEDS == tuple(range(52, 62))
    assert ConditionalProfitabilityConfig().validation_seeds == tuple(range(52, 62))


def test_config_requires_disjoint_development_and_validation_seeds() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        ConditionalProfitabilityConfig(
            development_seeds=(42, 52), validation_seeds=(52, 53)
        )


def test_config_rejects_changed_frozen_weights_and_bad_quantiles() -> None:
    with pytest.raises(ValueError, match="frozen"):
        ConditionalProfitabilityConfig(current_rack_weight=0.9)
    with pytest.raises(ValueError, match="ordered"):
        ConditionalProfitabilityConfig(score_quantiles=(0.3, 0.1, 0.7, 0.9))
    with pytest.raises(ValueError, match="inside"):
        ConditionalProfitabilityConfig(score_quantiles=(0.0, 0.3, 0.7, 0.9))


def test_cutpoints_accept_scores_only_and_are_frozen() -> None:
    scores = (-4.0, -2.0, 0.0, 2.0, 4.0)
    cutpoints = freeze_score_cutpoints(scores, (0.1, 0.3, 0.7, 0.9))
    assert tuple(cutpoints.as_dict().values()) == pytest.approx((-3.2, -1.6, 1.6, 3.2))
    mutable_scores = list(scores)
    mutable_scores[0] = -999.0
    assert tuple(cutpoints.as_dict().values()) == pytest.approx((-3.2, -1.6, 1.6, 3.2))
    assert not hasattr(cutpoints, "monetary_outcome")


def test_linear_quantile_interpolates_deterministically() -> None:
    assert linear_quantile((0.0, 10.0), 0.3) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-2.0, "strong_high_rich"),
        (-1.999, "moderate_high_rich"),
        (-1.0, "moderate_high_rich"),
        (-0.999, "neutral"),
        (1.0, "neutral"),
        (1.001, "moderate_low_rich"),
        (2.0, "moderate_low_rich"),
        (2.001, "strong_low_rich"),
    ],
)
def test_band_assignment_boundaries(score: float, expected: str) -> None:
    cutpoints = ScoreBandCutpoints(-2.0, -1.0, 1.0, 2.0)
    assert assign_score_band(score, cutpoints) == expected


def test_every_score_belongs_to_exactly_one_band() -> None:
    cutpoints = ScoreBandCutpoints(-2.0, -1.0, 1.0, 2.0)
    assigned = [assign_score_band(score / 10, cutpoints) for score in range(-100, 101)]
    assert len(assigned) == 201
    assert set(assigned) == set(SCORE_BANDS)


def test_frozen_score_is_calculated_before_draw_and_cannot_receive_round_outcome() -> (
    None
):
    config = ConditionalProfitabilityConfig(
        development_seeds=(42,),
        validation_seeds=(52,),
        development_rounds_per_seed=1,
        validation_rounds_per_seed=1,
        burn_in_rounds=0,
    )
    source, game, strategy = _new_trajectory(config, "physical_iid", 52)
    draw_count = source.draw_count
    rack = game.pending_discard_rack
    state = _frozen_state(config, source, game)
    frozen_score = state.predicted_hi_lo_shift
    assert source.draw_count == draw_count
    assert game.pending_discard_rack == rack

    game.play_round(round_index=0, card_source=source, strategy=strategy)
    assert state.predicted_hi_lo_shift == frozen_score


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [([3.0, 3.0, 3.0], 0.0), ([1.0, 2.0, 3.0], 1.0), ([-1.0, -2.0, -3.0], -1.0)],
)
def test_monetary_regression_signs(outcomes: list[float], expected: float) -> None:
    regression = RegressionDiagnostics()
    for score, outcome in zip((1.0, 2.0, 3.0), outcomes, strict=True):
        regression.add(score, outcome)
    assert regression.as_dict()["slope"] == pytest.approx(expected)


def test_paired_source_monetary_slope_difference() -> None:
    rows = [
        {"source": "physical_iid", "seed": 52, "slope": 0.5},
        {"source": "one2six", "seed": 52, "slope": -0.5},
        {"source": "physical_iid", "seed": 53, "slope": 0.25},
        {"source": "one2six", "seed": 53, "slope": -0.75},
    ]
    result = _paired_monetary_slopes(rows, (52, 53))[0]
    assert result["mean_difference"] == -1.0
    assert result["negative_seeds"] == 2


def test_band_accumulator_reconciles_money_events_and_composition() -> None:
    accumulator = BandAccumulator()
    accumulator.add(observation(net=10.0, outcome="win"))
    accumulator.add(observation(net=-20.0, outcome="loss"))
    accumulator.add(observation(net=0.0, outcome="push"))
    metrics = accumulator.as_metrics()
    assert metrics["rounds"] == 3
    assert metrics["initial_wager"] == 30.0
    assert metrics["additional_action_wager"] == 30.0
    assert metrics["total_wager"] == 60.0
    assert metrics["total_player_net"] == -10.0
    assert metrics["total_casino_net"] == 10.0
    assert metrics["winning_rounds"] == 1
    assert metrics["losing_rounds"] == 1
    assert metrics["push_rounds"] == 1
    assert metrics["player_blackjack_count"] == 3
    assert metrics["double_actions"] == 3
    assert metrics["split_actions"] == 3


def test_missing_band_is_excluded_not_faked() -> None:
    rows = [band_row("physical_iid", 52, "neutral", -0.01)]
    aggregate = _aggregate_band_metrics(rows)
    assert len(aggregate) == 1
    assert aggregate[0]["score_band"] == "neutral"


def test_pre_specified_contrasts_and_source_differences() -> None:
    rows = complete_band_rows("physical_iid", 52, 0.0)
    rows += complete_band_rows("one2six", 52, 0.1)
    contrasts = _score_band_contrasts(rows, (52,))
    one = next(
        row
        for row in contrasts
        if row["source"] == "one2six"
        and row["metric"] == "strong_high_rich_minus_neutral"
    )
    direct = next(
        row
        for row in contrasts
        if row["source"] == "one2six_minus_physical_iid"
        and row["metric"] == "strong_high_rich_minus_neutral"
    )
    assert one["mean"] == pytest.approx(0.2)
    assert direct["mean"] == pytest.approx(0.0)


def test_ordered_trend_uses_negative_player_edge_sign_for_favourable_high_rich() -> (
    None
):
    rows = complete_band_rows("physical_iid", 52, 0.0)
    rows += complete_band_rows("one2six", 52, 0.1)
    trends = _ordered_band_trends(rows, (52,))
    one = next(
        row
        for row in trends
        if row["row_scope"] == "per_seed" and row["source"] == "one2six"
    )
    assert one["player_edge_ordered_band_slope"] == pytest.approx(-0.1)
    assert one["advantage_direction_slope"] == pytest.approx(0.1)


def test_candidate_gate_all_conditions_satisfied() -> None:
    result = evaluate_candidate_state(
        one2six_band=candidate_band("one2six", mean=0.02, lower=0.01, positives=10),
        iid_band=candidate_band("physical_iid", mean=0.0, lower=-0.01, positives=5),
        paired_difference={"student_t_95_ci": [0.01, 0.03]},
        total_one2six_rounds=100_000,
    )
    assert result["candidate_positive_ev_state"] is True


@pytest.mark.parametrize(
    ("one_changes", "iid_changes", "difference_ci", "total", "failed_condition"),
    [
        (
            {"player_edge_per_initial_wager_seed_ci": [-0.01, 0.03]},
            {},
            [0.01, 0.03],
            100_000,
            "positive_seed_ci",
        ),
        (
            {"positive_seed_player_edge_per_initial_wager": 8},
            {},
            [0.01, 0.03],
            100_000,
            "at_least_nine_positive_seeds",
        ),
        (
            {"rounds": 4_999},
            {},
            [0.01, 0.03],
            100_000,
            "at_least_five_percent_frequency",
        ),
        (
            {},
            {
                "mean_seed_player_edge_per_initial_wager": 0.01,
                "player_edge_per_initial_wager_seed_ci": [0.001, 0.02],
            },
            [0.01, 0.03],
            100_000,
            "physical_iid_not_same_positive_effect",
        ),
        ({}, {}, [-0.01, 0.03], 100_000, "positive_paired_source_difference"),
    ],
)
def test_candidate_gate_rejects_each_failed_condition(
    one_changes: dict[str, object],
    iid_changes: dict[str, object],
    difference_ci: list[float],
    total: int,
    failed_condition: str,
) -> None:
    one = candidate_band("one2six", mean=0.02, lower=0.01, positives=10)
    iid = candidate_band("physical_iid", mean=0.0, lower=-0.01, positives=5)
    one.update(one_changes)
    iid.update(iid_changes)
    result = evaluate_candidate_state(
        one2six_band=one,
        iid_band=iid,
        paired_difference={"student_t_95_ci": difference_ci},
        total_one2six_rounds=total,
    )
    assert result["candidate_positive_ev_state"] is False
    assert result["conditions"][failed_condition] is False


def test_deterministic_permutation_preserves_values_but_changes_alignment() -> None:
    scores = tuple(float(value) for value in range(20))
    outcomes = tuple(value**2 for value in scores)
    first = deterministic_permutation_regression(scores, outcomes, seed=52)
    second = deterministic_permutation_regression(scores, outcomes, seed=52)
    assert first["permuted_scores"] == second["permuted_scores"]
    assert sorted(first["permuted_scores"]) == list(scores)
    assert first["permuted_scores"] != scores
    assert outcomes == tuple(value**2 for value in scores)
    assert first["sample_count"] == len(scores)


def test_decision_recorder_delegates_action_and_records_schema() -> None:
    counts: dict[tuple[str, ...], int] = {}
    base = PublishedApproxCasinoStrategy()
    recorder = DecisionFrequencyRecorder(base, counts, source="one2six", band="neutral")
    decision = BlackjackDecisionState(
        player_ranks=("T", "6"),
        dealer_upcard_rank="6",
        legal_actions=frozenset({ActionType.HIT, ActionType.STAND}),
        is_split_hand=False,
    )
    assert recorder.choose_action(decision=decision) == base.choose_action(
        decision=decision
    )
    key = next(iter(counts))
    assert key[0:2] == ("one2six", "neutral")
    assert key[3:5] == ("hard_total", "16")
    assert "hit" in key[-1] and "stand" in key[-1]


@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        (("A", "7"), ("soft_total", "18")),
        (("T", "6"), ("hard_total", "16")),
        (("T", "Q"), ("pair", "10")),
        (("A", "A"), ("pair", "A")),
    ],
)
def test_decision_hand_categories(
    ranks: tuple[str, ...], expected: tuple[str, str]
) -> None:
    assert decision_hand_category(ranks) == expected


def test_small_experiment_writes_required_private_free_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "conditional"
    summary = run_conditional_profitability_experiment(
        ConditionalProfitabilityConfig(
            development_seeds=(42,),
            validation_seeds=(52,),
            development_rounds_per_seed=200,
            validation_rounds_per_seed=500,
            burn_in_rounds=10,
            output_dir=output_dir,
        )
    )
    assert summary["hidden_state_exported"] is False
    assert summary["score_band_cutpoints"]["monetary_outcomes_used"] is False
    assert (
        sum(
            int(row["rounds"])
            for row in summary["per_seed_score_band_profitability"]
            if row["source"] == "one2six"
        )
        == 500
    )
    required = {
        "summary.json",
        "summary.md",
        "experiment_config.json",
        "score_band_cutpoints.json",
        "continuous_monetary_slopes.csv",
        "paired_source_monetary_slopes.csv",
        "permutation_placebo_slopes.csv",
        "score_band_profitability.csv",
        "per_seed_score_band_profitability.csv",
        "score_band_contrasts.csv",
        "ordered_band_trends.csv",
        "score_band_initial_deal_composition.csv",
        "score_band_event_rates.csv",
        "score_band_frequency.csv",
        "decision_state_frequency.csv",
    }
    assert required.issubset(path.name for path in output_dir.iterdir())
    exported = "".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.iterdir()
        if path.suffix in {".json", ".csv", ".md"}
    ).lower()
    assert not any(term in exported for term in PRIVATE_TERMS)
    decision_text = (output_dir / "decision_state_frequency.csv").read_text(
        encoding="utf-8"
    )
    assert "alternative" not in decision_text.lower()


def observation(*, net: float, outcome: str) -> RoundMonetaryObservation:
    low = CardIndicators(hi_lo=1, low=1, neutral=0, ten_value=0, ace=0)
    ten = CardIndicators(hi_lo=-1, low=0, neutral=0, ten_value=1, ace=0)
    ace = CardIndicators(hi_lo=-1, low=0, neutral=0, ten_value=0, ace=1)
    return RoundMonetaryObservation(
        score=-0.01,
        band="strong_high_rich",
        box_net=net,
        initial_wager=10.0,
        action_wager=10.0,
        total_wager=20.0,
        player_blackjacks=1,
        double_actions=1,
        split_actions=1,
        outcome=outcome,
        cards_consumed=5,
        initial_cards=(low, ten, ace),
    )


def band_row(source: str, seed: int, band: str, edge: float) -> dict[str, object]:
    accumulator = BandAccumulator()
    accumulator.add(observation(net=edge * 10.0, outcome="win" if edge > 0 else "loss"))
    return {
        "source": source,
        "seed": seed,
        "score_band": band,
        "band_order": BAND_SCORES[band],
        "round_frequency": 0.2,
        **accumulator.as_metrics(),
        "player_edge_per_initial_wager": edge,
    }


def complete_band_rows(
    source: str, seed: int, intercept: float
) -> list[dict[str, object]]:
    return [
        band_row(source, seed, band, intercept - 0.1 * BAND_SCORES[band])
        for band in SCORE_BANDS
    ]


def candidate_band(
    source: str, *, mean: float, lower: float, positives: int
) -> dict[str, object]:
    return {
        "source": source,
        "score_band": "strong_high_rich",
        "rounds": 10_000,
        "mean_seed_player_edge_per_initial_wager": mean,
        "player_edge_per_initial_wager_seed_ci": [lower, mean + 0.01],
        "positive_seed_player_edge_per_initial_wager": positives,
    }
