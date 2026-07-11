# SPDX-License-Identifier: GPL-3.0-or-later

import json
from copy import deepcopy
from dataclasses import asdict
from math import sqrt
from pathlib import Path

import pytest
from experiments.observable_card_response import (
    BASELINE_PROBABILITIES,
    PRIVATE_EXPORT_TERMS,
    ObservableCardResponseConfig,
    PrefixCategorySums,
    RecordingCardSource,
    RegressionAccumulator,
    _memory_horizon_rows,
    _seed_summary_fields,
    card_indicators,
    initial_deal_position_map,
    rack_features,
    run_observable_card_response_experiment,
)
from experiments.single_box_game_validation import student_t_summary

from shufflemaster_sim.card_sources import (
    One2SixCardSource,
    One2SixConfig,
    ScriptedCardSource,
)
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)


@pytest.mark.parametrize("rank", ["2", "3", "4", "5", "6"])
def test_low_card_classification(rank: Rank) -> None:
    indicators = card_indicators(rank)
    assert indicators.low == 1
    assert indicators.hi_lo == 1
    assert indicators.neutral == indicators.ten_value == indicators.ace == 0


@pytest.mark.parametrize("rank", ["7", "8", "9"])
def test_neutral_card_classification(rank: Rank) -> None:
    indicators = card_indicators(rank)
    assert indicators.neutral == 1
    assert indicators.hi_lo == 0
    assert indicators.low == indicators.ten_value == indicators.ace == 0


@pytest.mark.parametrize("rank", ["T", "J", "Q", "K"])
def test_ten_value_card_classification(rank: Rank) -> None:
    indicators = card_indicators(rank)
    assert indicators.ten_value == 1
    assert indicators.hi_lo == -1
    assert indicators.low == indicators.neutral == indicators.ace == 0


def test_ace_is_separate_from_ten_value() -> None:
    indicators = card_indicators("A")
    assert indicators.ace == 1
    assert indicators.ten_value == 0
    assert indicators.hi_lo == -1


def test_rack_feature_formulas_use_hand_calculated_counts() -> None:
    cards = [make_card("2", 0), make_card("7", 1), make_card("T", 2), make_card("A", 3)]

    features = rack_features(cards)

    assert features.rack_size == 4
    assert features.rack_low_count == 1
    assert features.rack_neutral_count == 1
    assert features.rack_ten_value_count == 1
    assert features.rack_ace_count == 1
    assert features.rack_hi_lo_count == -1
    assert features.remaining_card_count == 308
    assert features.finite_pool_expected_hi_lo == pytest.approx(1 / 308)
    assert features.finite_pool_low_probability == pytest.approx(119 / 308)
    assert features.finite_pool_neutral_probability == pytest.approx(71 / 308)
    assert features.finite_pool_ten_value_probability == pytest.approx(95 / 308)
    assert features.finite_pool_ace_probability == pytest.approx(23 / 308)
    assert features.finite_pool_low_shift == pytest.approx(
        119 / 308 - BASELINE_PROBABILITIES["low"]
    )
    assert features.finite_pool_neutral_shift == pytest.approx(
        71 / 308 - BASELINE_PROBABILITIES["neutral"]
    )
    assert features.finite_pool_ten_value_shift == pytest.approx(
        95 / 308 - BASELINE_PROBABILITIES["ten_value"]
    )
    assert features.finite_pool_ace_shift == pytest.approx(
        23 / 308 - BASELINE_PROBABILITIES["ace"]
    )


def test_fifteen_card_probe_keeps_rack_out_and_original_untouched() -> None:
    source = One2SixCardSource(config=One2SixConfig(deck_count=6), seed=42)
    game = CasinoBlackjackGame(CasinoBlackjackConfig(base_bet=10.0, box_count=1))
    game.play_round(
        round_index=0,
        card_source=source,
        strategy=PublishedApproxCasinoStrategy(),
    )
    visible_rack = game.pending_discard_rack
    accepted_before = source.accepted_discard_batch_count
    draw_count_before = source.draw_count
    probe_source = deepcopy(source)

    probe = [probe_source.draw_card() for _ in range(15)]

    assert source.draw_count == draw_count_before
    assert source.accepted_discard_batch_count == accepted_before
    assert probe_source.accepted_discard_batch_count == accepted_before
    assert game.pending_discard_rack == visible_rack
    probe_source.assert_invariants([*visible_rack, *probe])


@pytest.mark.parametrize("box_count", range(1, 8))
def test_probe_prefix_maps_every_box_count(box_count: int) -> None:
    mapping = initial_deal_position_map(box_count)
    assert mapping["valid_probe_prefix"] == 2 * box_count + 1
    assert mapping["dealer_upcard_position"] == box_count + 1
    assert mapping["player_first_card_positions"] == list(range(1, box_count + 1))
    assert mapping["player_second_card_positions"] == list(
        range(box_count + 2, 2 * box_count + 2)
    )
    all_positions = [
        *mapping["player_first_card_positions"],
        mapping["dealer_upcard_position"],
        *mapping["player_second_card_positions"],
    ]
    assert sorted(all_positions) == list(range(1, 2 * box_count + 2))


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([4.0, -3.0, 7.0, 2.0], 0.0),
        ([1.0, 2.0, 3.0, 4.0], 1.0),
        ([-1.0, -2.0, -3.0, -4.0], -1.0),
    ],
)
def test_synthetic_regression_slopes(outcomes: list[float], expected: float) -> None:
    accumulator = RegressionAccumulator()
    predictors = [1.0, 2.0, 3.0, 4.0]
    if expected == 0.0:
        outcomes = [5.0, 5.0, 5.0, 5.0]
    for predictor, outcome in zip(predictors, outcomes, strict=True):
        accumulator.add(predictor, outcome)
    assert accumulator.slope() == pytest.approx(expected)


def test_zero_predictor_variance_returns_missing_slope() -> None:
    accumulator = RegressionAccumulator()
    accumulator.add(1.0, 2.0)
    accumulator.add(1.0, 3.0)
    assert accumulator.slope() is None


def test_prefix_sums_exact_lags_bands_and_censoring() -> None:
    draws = {
        "hi_lo": [1, 0, -1, 1, 1, 0, -1, -1],
        "low": [1, 0, 0, 1, 1, 0, 0, 0],
        "neutral": [0, 1, 0, 0, 0, 1, 0, 0],
        "ten_value": [0, 0, 1, 0, 0, 0, 1, 0],
        "ace": [0, 0, 0, 0, 0, 0, 0, 1],
    }
    prefixes = PrefixCategorySums.from_draws(draws)

    assert (
        prefixes.window_mean("hi_lo", return_draw_index=2, start_lag=1, end_lag=1) == -1
    )
    assert prefixes.window_mean(
        "hi_lo", return_draw_index=2, start_lag=1, end_lag=3
    ) == pytest.approx(1 / 3)
    assert prefixes.window_mean(
        "low", return_draw_index=2, start_lag=2, end_lag=4
    ) == pytest.approx(2 / 3)
    assert prefixes.window_mean("ace", return_draw_index=2, start_lag=6, end_lag=6) == 1
    assert (
        prefixes.window_mean("hi_lo", return_draw_index=2, start_lag=1, end_lag=7)
        is None
    )


def test_return_index_is_first_future_draw_and_batch_is_observable_only() -> None:
    source = ScriptedCardSource([("2", "spades"), ("T", "clubs"), ("A", "hearts")])
    wrapped = RecordingCardSource(source=source, source_name="physical_iid", seed=42)
    wrapped.enabled = True
    first = wrapped.draw_card()
    second = wrapped.draw_card()

    wrapped.accept_discards([first, second])
    event = wrapped.batches[0]

    assert event.return_draw_index == 2
    assert event.features.rack_size == 2
    assert event.features.rack_hi_lo_count == 0
    assert source.accepted_discard_batches == [[first, second]]
    exported = json.dumps(asdict(event)).lower()
    assert not any(term in exported for term in PRIVATE_EXPORT_TERMS)


def test_seed_slope_aggregation_includes_uncertainty_and_signs() -> None:
    slopes = [1.0, 2.0, 3.0]
    fields = _seed_summary_fields(student_t_summary(slopes), slopes)

    assert fields["contributing_seeds"] == 3
    assert fields["mean_seed_slope"] == 2.0
    assert fields["sample_standard_deviation"] == 1.0
    assert fields["standard_error"] == pytest.approx(1 / sqrt(3))
    assert fields["positive_seed_slopes"] == 3
    assert fields["negative_seed_slopes"] == 0


def test_memory_horizon_requires_all_later_bands_to_be_negligible() -> None:
    rows = lag_rows(
        [
            ("1-15", 0.50, 0.50, [0.20, 0.80]),
            ("16-50", 0.05, 0.05, [-0.10, 0.20]),
            ("51-100", 0.20, 0.20, [-0.05, 0.45]),
            ("101-250", 0.04, 0.04, [-0.10, 0.18]),
            ("251-500", -0.03, 0.03, [-0.12, 0.06]),
            ("501-1000", 0.02, 0.02, [-0.08, 0.12]),
        ]
    )

    result = _memory_horizon_rows(rows, threshold=0.10)
    hi_lo = next(
        row
        for row in result
        if row["source"] == "one2six" and row["outcome"] == "hi_lo"
    )
    assert hi_lo["earliest_lag_all_later_negligible"] == "101-250"
    assert hi_lo["latest_lag_clearly_detected"] == "1-15"
    assert hi_lo["response_changes_sign"] is True


def test_non_monotonic_later_detection_prevents_early_horizon() -> None:
    rows = lag_rows(
        [
            ("1-15", 0.05, 0.05, [-0.10, 0.20]),
            ("16-50", 0.40, 0.40, [0.20, 0.60]),
            ("51-100", 0.02, 0.02, [-0.10, 0.14]),
        ]
    )
    result = _memory_horizon_rows(rows, threshold=0.10)
    hi_lo = next(
        row
        for row in result
        if row["source"] == "one2six" and row["outcome"] == "hi_lo"
    )
    assert hi_lo["earliest_lag_all_later_negligible"] == "51-100"


def test_small_experiment_exports_no_hidden_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "response"
    config = ObservableCardResponseConfig(
        seeds=(42,),
        current_rack_states_per_seed=3,
        current_rack_burn_in_rounds=2,
        current_rack_sample_interval_rounds=1,
        lag_rounds_per_seed=300,
        lag_burn_in_rounds=2,
        output_dir=output_dir,
    )

    summary = run_observable_card_response_experiment(config)

    assert summary["hidden_state_exported"] is False
    expected = {
        "summary.json",
        "summary.md",
        "experiment_config.json",
        "current_rack_primary_response.csv",
        "current_rack_position_response.csv",
        "current_rack_per_seed_slopes.csv",
        "current_rack_role_summary.csv",
        "current_rack_state_frequency.csv",
        "returned_batch_lag_response.csv",
        "returned_batch_exact_lag_response.csv",
        "returned_batch_per_seed_slopes.csv",
        "returned_batch_window_counts.csv",
        "memory_horizon_summary.csv",
        "memory_horizon_overview.png",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    public_text = "".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.iterdir()
        if path.suffix in {".json", ".csv", ".md"}
    ).lower()
    assert not any(term in public_text for term in PRIVATE_EXPORT_TERMS)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="six decks"):
        ObservableCardResponseConfig(deck_count=8)
    with pytest.raises(ValueError, match="equal 15"):
        ObservableCardResponseConfig(current_rack_probe_cards=14)
    with pytest.raises(ValueError, match="at least 1,000"):
        ObservableCardResponseConfig(lag_horizon_cards=999)
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        ObservableCardResponseConfig(lag_bands=((1, 15), (15, 20)))


def make_card(rank: Rank, index: int) -> Card:
    return Card(
        rank=rank,
        suit="spades",
        physical_id=f"test:{index}",
        draw_id=index,
    )


def lag_rows(
    values: list[tuple[str, float, float, list[float]]],
) -> list[dict[str, object]]:
    return [
        {
            "source": "one2six",
            "lag": lag,
            "outcome": "hi_lo",
            "mean_seed_slope": mean,
            "mean_absolute_seed_slope": mean_absolute,
            "student_t_95_ci": interval,
            "contributing_seeds": 5,
        }
        for lag, mean, mean_absolute, interval in values
    ]
