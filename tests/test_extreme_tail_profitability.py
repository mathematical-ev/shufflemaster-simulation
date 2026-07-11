# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy

import pytest
from experiments.extreme_tail_profitability import (
    ExtremeTailProfitabilityConfig,
    InsuranceSnapshot,
    TailCutpoints,
    decline_even_money_value,
    disjoint_slice,
    freeze_cutpoints,
    insurance_ev,
    insurance_gate,
    insurance_second_card,
    nested_sets,
    planning_precision,
    positive_ev_gate,
)
from experiments.fading_exclusion_validation import (
    LedgerCardSource,
    ObservableCohortLedger,
)
from experiments.multi_box_counterfactual import _make_source

from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)


def simple_cutpoints() -> TailCutpoints:
    return TailCutpoints(
        {
            "q1": 1,
            "q2_5": 2,
            "q5": 3,
            "q10": 4,
            "q20": 5,
            "q40": 6,
            "q60": 7,
            "q80": 8,
            "q90": 9,
            "q95": 10,
            "q97_5": 11,
            "q99": 12,
        }
    )


def test_cutpoints_depend_only_on_scores() -> None:
    scores = list(range(100))
    frozen = freeze_cutpoints(scores)
    before = dict(frozen.values)
    monetary_outcomes = [999.0] * 100
    monetary_outcomes.reverse()
    assert dict(frozen.values) == before
    assert not hasattr(frozen, "monetary_outcomes")


def test_nested_assignments_overlap_and_boundaries_are_inclusive() -> None:
    cutpoints = simple_cutpoints()
    assert nested_sets(1, cutpoints) == (
        "high_rich_20",
        "high_rich_10",
        "high_rich_05",
        "high_rich_025",
        "high_rich_01",
    )
    assert nested_sets(6, cutpoints) == ("neutral",)
    assert nested_sets(12, cutpoints) == (
        "low_rich_20",
        "low_rich_10",
        "low_rich_05",
        "low_rich_025",
        "low_rich_01",
    )


def test_disjoint_slice_boundary_assignment() -> None:
    cutpoints = simple_cutpoints()
    assert disjoint_slice(1, cutpoints) == "lowest_0_1"
    assert disjoint_slice(1.1, cutpoints) == "1_2_5"
    assert disjoint_slice(12, cutpoints) == "97_5_99"
    assert disjoint_slice(12.1, cutpoints) == "99_100"


def test_insurance_and_even_money_break_even_formulas() -> None:
    assert insurance_ev(1 / 3, 3) == pytest.approx(0)
    assert insurance_ev(1 / 11, 11) == pytest.approx(0)
    assert decline_even_money_value(1 / 3) == pytest.approx(1.0)
    assert decline_even_money_value(0.2) > 1.0


def gate_row(mean=0.01, low=0.001, positive=8, frequency=0.01):
    return {
        "mean": mean,
        "student_t_95_ci": [low, mean + 0.01],
        "positive_seeds": positive,
        "frequency": frequency,
    }


def test_positive_ev_gate_passes_and_fails_each_requirement() -> None:
    iid = gate_row(mean=-0.01, low=-0.02, positive=2)
    assert positive_ev_gate(gate_row(), iid, gate_row(), gate_row())
    assert not positive_ev_gate(gate_row(mean=-0.01), iid, gate_row(), gate_row())
    assert not positive_ev_gate(gate_row(low=-0.01), iid, gate_row(), gate_row())
    assert not positive_ev_gate(gate_row(positive=7), iid, gate_row(), gate_row())
    assert not positive_ev_gate(gate_row(frequency=0.004), iid, gate_row(), gate_row())
    assert not positive_ev_gate(gate_row(), gate_row(), gate_row(), gate_row())


def test_insurance_gate_requires_margin_support_and_negative_control() -> None:
    row = {
        "mean_probability": 0.36,
        "student_t_95_ci": [0.34, 0.38],
        "seeds_above_threshold": 8,
        "minimum_seed_opportunities": 100,
    }
    iid = {
        "mean_probability": 0.31,
        "student_t_95_ci": [0.30, 0.32],
    }
    assert insurance_gate(row, iid, threshold=1 / 3)
    changed = dict(row, minimum_seed_opportunities=99)
    assert not insurance_gate(changed, iid, threshold=1 / 3)


def test_precision_planning_scales_with_target() -> None:
    result = planning_precision([0.01, 0.02, 0.00, 0.015, 0.005], 10_000)
    assert result["approximate_mde_80_percent_power"] > 0
    assert result["approximate_rounds_for_25bp"] > result["approximate_rounds_for_50bp"]
    assert (
        result["approximate_rounds_for_50bp"] > result["approximate_rounds_for_100bp"]
    )


def test_no_hole_card_branch_is_deterministic_and_isolated() -> None:
    game = CasinoBlackjackGame(CasinoBlackjackConfig())
    source = LedgerCardSource(
        _make_source("physical_iid", 6, 91), ObservableCohortLedger()
    )
    source.before_round()
    game.burn_initial_card(source)
    table = game.create_table(0)
    game.deal_initial_cards(table, source)
    snapshot = InsuranceSnapshot(game, table, source)
    original = deepcopy(snapshot)
    strategy = PublishedApproxCasinoStrategy()
    first = insurance_second_card(snapshot, strategy)
    second = insurance_second_card(snapshot, strategy)
    assert first == second
    assert snapshot.table == original.table
    assert snapshot.source.draw_count == original.source.draw_count


def test_config_preserves_frozen_weights_and_seed_isolation() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        ExtremeTailProfitabilityConfig(
            development_seeds=(82, 87), validation_seeds=(87, 88)
        )
    with pytest.raises(ValueError, match="frozen"):
        ExtremeTailProfitabilityConfig(current_rack_weight=0.9)
