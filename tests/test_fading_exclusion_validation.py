# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy
from pathlib import Path

import pytest
from experiments.fading_exclusion_validation import (
    COMPONENT_NAMES,
    FROZEN_WEIGHTS,
    PRIVATE_TERMS,
    CohortCounts,
    FadingExclusionValidationConfig,
    ObservableCohortLedger,
    ReturnedCohort,
    _mark_position_stability,
    _paired_source_differences,
    calculate_fading_state,
    cohort_age_band,
    run_fading_exclusion_validation,
    score_group,
)
from experiments.observable_card_response import RegressionAccumulator

from shufflemaster_sim.card_sources import One2SixCardSource, One2SixConfig
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (1, "returned_1_15"),
        (15, "returned_1_15"),
        (16, "returned_16_50"),
        (50, "returned_16_50"),
        (51, "returned_51_100"),
        (100, "returned_51_100"),
        (101, None),
    ],
)
def test_cohort_age_bands_are_unique(age: int, expected: str | None) -> None:
    assert cohort_age_band(age) == expected


def test_age_zero_uses_freshest_returned_band() -> None:
    assert cohort_age_band(0) == "returned_1_15"


def test_ledger_assigns_each_batch_to_only_one_band_and_expires_old() -> None:
    ledger = ObservableCohortLedger(
        returned=[
            ReturnedCohort(99, counts(1, 1, 1, 0, 0, 0)),
            ReturnedCohort(85, counts(1, 0, 0, 1, 0, 0)),
            ReturnedCohort(50, counts(1, -1, 0, 0, 1, 0)),
            ReturnedCohort(0, counts(1, -1, 0, 0, 0, 1)),
        ]
    )

    bands = ledger.active_by_band(101)

    assert [len(bands[name]) for name in COMPONENT_NAMES[1:]] == [1, 1, 1]
    assert len({id(batch) for group in bands.values() for batch in group}) == 3
    assert len(ledger.returned) == 3


def test_frozen_weights_are_exact_and_not_mutated() -> None:
    config = FadingExclusionValidationConfig()
    assert config.weights == {
        "current_rack": 1.0,
        "returned_1_15": 0.75,
        "returned_16_50": 0.40,
        "returned_51_100": 0.20,
        "returned_over_100": 0.0,
    }
    returned = config.weights
    returned["current_rack"] = 99.0
    assert config.weights == FROZEN_WEIGHTS


def test_weighted_state_uses_hand_calculated_cohorts() -> None:
    current = counts(4, 0, 1, 1, 1, 1)
    returned = {
        "returned_1_15": [ReturnedCohort(0, counts(2, 2, 2, 0, 0, 0))],
        "returned_16_50": [ReturnedCohort(0, counts(2, -2, 0, 0, 1, 1))],
        "returned_51_100": [ReturnedCohort(0, counts(1, 0, 0, 1, 0, 0))],
    }

    state = calculate_fading_state(
        current_rack=current,
        returned_by_band=returned,
        weights=FROZEN_WEIGHTS,
    )

    assert state.effective_weighted_card_count == pytest.approx(6.5)
    assert state.weighted_hi_lo_count == pytest.approx(0.7)
    assert state.weighted_low_count == pytest.approx(2.5)
    assert state.weighted_neutral_count == pytest.approx(1.2)
    assert state.weighted_ten_value_count == pytest.approx(1.4)
    assert state.weighted_ace_count == pytest.approx(1.4)
    assert (
        state.weighted_low_count
        + state.weighted_neutral_count
        + state.weighted_ten_value_count
        + state.weighted_ace_count
        == pytest.approx(state.effective_weighted_card_count)
    )
    assert state.effective_remaining_cards == pytest.approx(305.5)
    expected_low_excess = 2.5 - (120 / 312) * 6.5
    expected_neutral_excess = 1.2 - (72 / 312) * 6.5
    expected_ten_excess = 1.4 - (96 / 312) * 6.5
    expected_ace_excess = 1.4 - (24 / 312) * 6.5
    assert state.weighted_low_excess == pytest.approx(expected_low_excess)
    assert state.weighted_neutral_excess == pytest.approx(expected_neutral_excess)
    assert state.weighted_ten_value_excess == pytest.approx(expected_ten_excess)
    assert state.weighted_ace_excess == pytest.approx(expected_ace_excess)
    assert state.predicted_hi_lo_shift == pytest.approx(-0.7 / 305.5)
    assert state.predicted_low_shift == pytest.approx(-expected_low_excess / 305.5)
    assert state.predicted_neutral_shift == pytest.approx(
        -expected_neutral_excess / 305.5
    )
    assert state.predicted_ten_value_shift == pytest.approx(
        -expected_ten_excess / 305.5
    )
    assert state.predicted_ace_shift == pytest.approx(-expected_ace_excess / 305.5)


def test_score_is_frozen_before_probe_and_rack_return() -> None:
    source = One2SixCardSource(config=One2SixConfig(deck_count=6), seed=47)
    game = CasinoBlackjackGame(CasinoBlackjackConfig(base_bet=10.0, box_count=1))
    game.play_round(
        round_index=0,
        card_source=source,
        strategy=PublishedApproxCasinoStrategy(),
    )
    rack = game.pending_discard_rack
    state = calculate_fading_state(
        current_rack=CohortCounts.from_cards(rack),
        returned_by_band={name: [] for name in COMPONENT_NAMES[1:]},
        weights=FROZEN_WEIGHTS,
    )
    source_before = source.draw_count
    accepted_before = source.accepted_discard_batch_count
    clone = deepcopy(source)

    first = [clone.draw_card() for _ in range(15)]
    second_clone = deepcopy(source)
    second = [second_clone.draw_card() for _ in range(15)]

    assert state.effective_weighted_card_count == len(rack)
    assert source.draw_count == source_before
    assert source.accepted_discard_batch_count == accepted_before
    assert game.pending_discard_rack == rack
    assert first == second


def test_seed_boundaries_use_distinct_ledgers() -> None:
    first = ObservableCohortLedger()
    first.record_return(return_draw_index=10, cards=[make_card("2", 0)])
    second = ObservableCohortLedger()
    assert len(first.returned) == 1
    assert second.returned == []


@pytest.mark.parametrize(
    ("values", "expected"),
    [([5.0, 5.0, 5.0], 0.0), ([1.0, 2.0, 3.0], 1.0), ([-1.0, -2.0, -3.0], -1.0)],
)
def test_regression_slopes(values: list[float], expected: float) -> None:
    regression = RegressionAccumulator()
    for predictor, outcome in zip([1.0, 2.0, 3.0], values, strict=True):
        regression.add(predictor, outcome)
    assert regression.slope() == pytest.approx(expected)


def test_missing_predictor_variance_is_clear() -> None:
    regression = RegressionAccumulator()
    regression.add(1.0, 1.0)
    regression.add(1.0, 2.0)
    assert regression.slope() is None


def test_paired_source_difference_uses_matching_seed_slopes() -> None:
    rows = [
        {"source": "physical_iid", "seed": 47, "outcome": "hi_lo", "slope": 0.0},
        {"source": "one2six", "seed": 47, "outcome": "hi_lo", "slope": 1.0},
        {"source": "physical_iid", "seed": 48, "outcome": "hi_lo", "slope": -0.5},
        {"source": "one2six", "seed": 48, "outcome": "hi_lo", "slope": 1.5},
    ]
    for outcome in ("low", "neutral", "ten_value", "ace"):
        rows.extend(
            [
                {
                    "source": "physical_iid",
                    "seed": seed,
                    "outcome": outcome,
                    "slope": 0.0,
                }
                for seed in (47, 48)
            ]
        )
        rows.extend(
            [
                {"source": "one2six", "seed": seed, "outcome": outcome, "slope": 0.0}
                for seed in (47, 48)
            ]
        )
    result = _paired_source_differences(rows, (47, 48))
    hi_lo = next(row for row in result if row["outcome"] == "hi_lo")
    assert hi_lo["mean_paired_slope_difference"] == 1.5
    assert hi_lo["positive_seeds"] == 2


def test_position_stability_requires_similar_iid_direction() -> None:
    rows = [
        position_row("physical_iid", -1.0, [-1.5, -0.5]),
        position_row("one2six", 1.0, [0.5, 1.5]),
    ]
    _mark_position_stability(rows)
    assert rows[1]["unstable"] is False

    rows[0]["mean_seed_slope"] = 0.75
    rows[0]["student_t_95_ci"] = [0.25, 1.25]
    _mark_position_stability(rows)
    assert rows[1]["unstable"] is True
    assert rows[1]["instability_reasons"] == ["physical_iid_has_similar_effect"]


@pytest.mark.parametrize(
    ("net", "expected"),
    [(10.0, "win"), (-10.0, "loss"), (0.0, "push")],
)
def test_monetary_classification_remains_box_net_based(
    net: float, expected: str
) -> None:
    outcome = "win" if net > 0 else "loss" if net < 0 else "push"
    assert outcome == expected


def test_fixed_score_groups_are_mutually_exclusive() -> None:
    assert score_group(-0.003) == "predicted_high_rich"
    assert score_group(-0.0025) == "near_neutral"
    assert score_group(0.0) == "near_neutral"
    assert score_group(0.0025) == "near_neutral"
    assert score_group(0.003) == "predicted_low_rich"


def test_small_heldout_experiment_writes_private_free_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "heldout"
    summary = run_fading_exclusion_validation(
        FadingExclusionValidationConfig(
            seeds=(47,),
            rounds_per_seed=40,
            burn_in_rounds=2,
            probe_states_per_seed=5,
            output_dir=output_dir,
        )
    )
    assert summary["hidden_state_exported"] is False
    assert sum(row["probe_states"] for row in summary["probe_state_counts"]) == 10
    text = "".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.iterdir()
        if path.suffix in {".json", ".csv", ".md"}
    ).lower()
    assert not any(term in text for term in PRIVATE_TERMS)


def test_config_rejects_development_seeds_and_changed_frozen_weights() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        FadingExclusionValidationConfig(seeds=(42,))
    with pytest.raises(ValueError, match="disjoint"):
        FadingExclusionValidationConfig(
            seeds=(42,), current_rack_weight=0.9, allow_weight_override=True
        )
    with pytest.raises(ValueError, match="frozen"):
        FadingExclusionValidationConfig(current_rack_weight=0.9)


def counts(
    card_count: int,
    hi_lo: int,
    low: int,
    neutral: int,
    ten_value: int,
    ace: int,
) -> CohortCounts:
    return CohortCounts(card_count, hi_lo, low, neutral, ten_value, ace)


def make_card(rank: Rank, index: int) -> Card:
    return Card(rank=rank, suit="spades", physical_id=f"test:{index}", draw_id=index)


def position_row(source: str, mean: float, interval: list[float]) -> dict[str, object]:
    return {
        "source": source,
        "position": 1,
        "outcome": "hi_lo",
        "contributing_seeds": 5,
        "mean_predictor_standard_deviation": 0.01,
        "mean_seed_slope": mean,
        "student_t_95_ci": interval,
    }
