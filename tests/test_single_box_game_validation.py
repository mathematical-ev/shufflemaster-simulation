# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import pytest
from experiments.metrics import MonetaryStreakTracker, classify_monetary_outcome
from experiments.single_box_game_validation import (
    PHYSICAL_IID_PAIR_OPPORTUNITY_RATE,
    PairActionRecorder,
    RunningMoments,
    SingleBoxGameValidationConfig,
    aggregate_streak_metrics,
    box_round_net_result,
    initial_pair_category,
    paired_difference_summary,
    run_single_box_game_validation,
    split_count_decomposition,
    student_t_summary,
)

from shufflemaster_sim.actions import ActionType, GameAction
from shufflemaster_sim.card_sources import One2SixCardSource, PhysicalIidCardSource
from shufflemaster_sim.cards import Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.state import (
    BlackjackDecisionState,
    BoxState,
    DealerState,
    HandState,
    TableState,
)
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rounds": 0}, "rounds must be positive"),
        ({"base_bet": 0.0}, "base_bet must be positive"),
        ({"deck_count": 8}, "exactly six decks"),
        ({"seeds": ()}, "at least one seed"),
    ],
)
def test_validation_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SingleBoxGameValidationConfig(**kwargs)


def test_small_validation_runs_both_sources_with_shared_profile(
    tmp_path: Path,
) -> None:
    summary = run_single_box_game_validation(
        SingleBoxGameValidationConfig(
            rounds=50,
            base_bet=10.0,
            seeds=(42, 43),
            output_dir=tmp_path,
        )
    )

    physical = summary["aggregate"]["physical_iid"]
    one2six = summary["aggregate"]["one2six"]
    assert physical["completed_rounds"] == one2six["completed_rounds"] == 100
    assert physical["deck_count"] == one2six["deck_count"] == 6
    assert physical["strategy_implementation"] == one2six["strategy_implementation"]
    assert one2six["one2six_invariant_passed"] is True
    assert physical["initial_wagered"] == one2six["initial_wagered"] == 1000.0
    assert physical["total_cards_drawn"] > 0
    assert one2six["total_cards_drawn"] > 0
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "per_seed_summary.csv").is_file()
    assert (tmp_path / "pair_summary.csv").is_file()
    assert (tmp_path / "pair_by_upcard.csv").is_file()
    assert (tmp_path / "streak_summary.csv").is_file()
    assert (tmp_path / "streak_length_distribution.csv").is_file()
    assert (tmp_path / "signed_streak_length_histogram.png").is_file()
    assert (tmp_path / "physical_iid_seed_42.json").is_file()
    assert (tmp_path / "one2six_seed_42.json").is_file()
    assert (
        summary["seed_level_uncertainty"]["sources"]["physical_iid"][
            "player_edge_per_initial_wager"
        ]["student_t_95_ci"]
        is not None
    )
    assert summary["theory"]["physical_iid_initial_pair_opportunity_rate"] == (
        PHYSICAL_IID_PAIR_OPPORTUNITY_RATE
    )
    for source_name in ("physical_iid", "one2six"):
        streaks = summary["aggregate"][source_name]["streak_metrics"]
        assert (
            streaks["winning_rounds"]
            + streaks["losing_rounds"]
            + streaks["push_rounds"]
            == streaks["rounds"]
        )


def test_sources_are_configured_with_six_decks() -> None:
    physical = PhysicalIidCardSource(deck_count=6, seed=42)
    one2six = One2SixCardSource(seed=42)

    assert physical.deck_count == 6
    assert one2six.config.deck_count == 6


def test_one2six_receives_previous_round_rack_after_next_initial_deal() -> None:
    source = One2SixCardSource(seed=42)
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            deck_count=6,
            dealer_hits_soft_17=False,
            burn_initial_card=True,
        )
    )
    strategy = PublishedApproxCasinoStrategy()

    first = game.play_round(round_index=0, card_source=source, strategy=strategy)
    assert len(source.accepted_discard_batches) == 1

    game.play_round(round_index=1, card_source=source, strategy=strategy)

    assert source.accepted_discard_batches[1] == first.discard_rack
    source.assert_invariants(external_cards=game.pending_discard_rack)


@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        (("A", "A"), "A-A"),
        (("8", "8"), "8-8"),
        (("T", "K"), "ten-value pair"),
        (("Q", "J"), "ten-value pair"),
        (("9", "T"), None),
    ],
)
def test_initial_pair_categories(
    ranks: tuple[Rank, ...],
    expected: str | None,
) -> None:
    assert initial_pair_category(ranks, is_split_hand=False) == expected


def test_split_created_hand_is_not_an_initial_pair_opportunity() -> None:
    assert initial_pair_category(("8", "8"), is_split_hand=True) is None


def decision(
    ranks: tuple[Rank, ...],
    dealer: Rank,
    legal_actions: set[ActionType],
    *,
    is_split_hand: bool = False,
) -> BlackjackDecisionState:
    return BlackjackDecisionState(
        player_ranks=ranks,
        dealer_upcard_rank=dealer,
        legal_actions=frozenset(legal_actions),
        is_split_hand=is_split_hand,
    )


def test_pair_actions_and_upcards_are_decomposed_and_reconciled() -> None:
    recorder = PairActionRecorder(PublishedApproxCasinoStrategy())
    recorder.choose_action(
        decision=decision(
            ("A", "A"),
            "A",
            {ActionType.HIT, ActionType.STAND, ActionType.SPLIT},
        )
    )
    recorder.choose_action(
        decision=decision(("4", "4"), "2", {ActionType.HIT, ActionType.SPLIT})
    )
    recorder.choose_action(
        decision=decision(
            ("T", "K"),
            "6",
            {ActionType.HIT, ActionType.STAND, ActionType.SPLIT},
        )
    )
    recorder.choose_action(
        decision=decision(
            ("5", "5"),
            "6",
            {ActionType.HIT, ActionType.DOUBLE, ActionType.SPLIT},
        )
    )

    metrics = recorder.as_metrics(rounds=4)

    assert metrics["initial_pair_opportunities"] == 4
    assert metrics["actual_split_actions"] == 1
    assert metrics["initial_pairs_not_split"] == 3
    assert metrics["non_split_actions"] == {"hit": 1, "stand": 1, "double": 1}
    assert metrics["categories"]["A-A"]["actual_splits"] == 1
    assert (
        metrics["categories"]["4-4"]["by_upcard"]["2"]["non_split_actions"]["hit"] == 1
    )
    assert (
        metrics["categories"]["ten-value pair"]["by_upcard"]["6"]["non_split_actions"][
            "stand"
        ]
        == 1
    )
    assert (
        sum(cell["opportunities"] for cell in metrics["categories"].values())
        == metrics["initial_pair_opportunities"]
    )
    assert (
        sum(
            upcard_cell["opportunities"]
            for cell in metrics["categories"].values()
            for upcard_cell in cell["by_upcard"].values()
        )
        == metrics["initial_pair_opportunities"]
    )


def test_expected_split_missing_from_legal_actions_is_diagnostic() -> None:
    recorder = PairActionRecorder(PublishedApproxCasinoStrategy())

    action = recorder.choose_action(
        decision=decision(("8", "8"), "T", {ActionType.HIT, ActionType.STAND})
    )
    metrics = recorder.as_metrics(rounds=1)

    assert action.action_type == ActionType.HIT
    assert metrics["expected_split_missing_legal_action"] == 1
    assert metrics["actual_split_actions"] == 0


class _AlwaysSplitStrategy:
    def choose_action(self, *, decision: BlackjackDecisionState) -> GameAction:
        _ = decision
        return GameAction(ActionType.SPLIT)


def test_actual_split_without_initial_pair_is_detected() -> None:
    recorder = PairActionRecorder(_AlwaysSplitStrategy())

    recorder.choose_action(decision=decision(("8", "7"), "6", {ActionType.SPLIT}))

    assert recorder.actual_split_actions == 1
    assert recorder.actual_split_without_initial_pair == 1


def test_running_round_profit_statistics() -> None:
    moments = RunningMoments()
    for value in (1.0, 2.0, 3.0):
        moments.add(value)

    statistics = moments.as_dict()

    assert statistics["count"] == 3
    assert statistics["sum"] == 6.0
    assert statistics["mean_round_profit"] == 2.0
    assert statistics["sample_variance_round_profit"] == 1.0
    assert statistics["sample_standard_deviation_round_profit"] == 1.0
    assert statistics["naive_round_se"] == pytest.approx(1.0 / (3.0**0.5))
    assert statistics["naive_normal_95_ci"] == pytest.approx(
        [0.8684142659, 3.1315857341]
    )


def test_student_t_seed_summary_uses_small_sample_critical_value() -> None:
    summary = student_t_summary([0.1, 0.2, 0.3])

    assert summary["mean"] == pytest.approx(0.2)
    assert summary["sample_standard_deviation"] == pytest.approx(0.1)
    assert summary["standard_error"] == pytest.approx(0.1 / (3.0**0.5))
    assert summary["student_t_critical"] == pytest.approx(4.30265273)
    assert summary["student_t_95_ci"] == pytest.approx([-0.0484137712, 0.4484137712])


def test_paired_seed_difference_summary_counts_signs() -> None:
    physical = [
        {"seed": 1, "edge": 0.01},
        {"seed": 2, "edge": 0.02},
        {"seed": 3, "edge": 0.03},
    ]
    one2six = [
        {"seed": 1, "edge": 0.02},
        {"seed": 2, "edge": 0.01},
        {"seed": 3, "edge": 0.03},
    ]

    summary = paired_difference_summary(physical, one2six, "edge")

    assert summary["mean"] == pytest.approx(0.0)
    assert summary["positive_differences"] == 1
    assert summary["negative_differences"] == 1
    assert summary["zero_differences"] == 1
    assert summary["differences_by_seed"] == {"1": 0.01, "2": -0.01, "3": 0.0}


def test_split_count_decomposition_reconciles_to_observed_difference() -> None:
    physical = {
        "initial_pair_opportunities": 1000,
        "split_actions": 200,
        "split_rate_given_initial_pair": 0.2,
    }
    one2six = {
        "initial_pair_opportunities": 900,
        "split_actions": 135,
        "split_rate_given_initial_pair": 0.15,
    }

    decomposition = split_count_decomposition(physical, one2six)

    assert decomposition["actual_split_count_difference"] == -65
    assert decomposition[
        "opportunity_volume_component_at_iid_conditional_rate"
    ] == pytest.approx(-20.0)
    assert decomposition[
        "conditional_mix_component_at_one2six_pair_count"
    ] == pytest.approx(-45.0)
    assert decomposition["component_sum"] == pytest.approx(-65.0)


def settled_box_table(hand_results: list[float]) -> TableState:
    return TableState(
        boxes=[
            BoxState(
                box_id=1,
                base_bet=10.0,
                hands=[
                    HandState(
                        hand_id=index,
                        cards=[],
                        wager=10.0,
                        is_split_hand=len(hand_results) > 1,
                        outcome_label="settled",
                        net_result=net_result,
                    )
                    for index, net_result in enumerate(hand_results)
                ],
            )
        ],
        dealer=DealerState(),
        round_index=0,
    )


@pytest.mark.parametrize(
    ("hand_results", "expected_net", "expected_outcome"),
    [
        ([10.0, -10.0], 0.0, "push"),
        ([20.0, -10.0], 10.0, "win"),
        ([10.0, -20.0], -10.0, "loss"),
        ([15.0], 15.0, "win"),
    ],
)
def test_box_round_classification_uses_combined_monetary_result(
    hand_results: list[float],
    expected_net: float,
    expected_outcome: str,
) -> None:
    net_result = box_round_net_result(settled_box_table(hand_results))

    assert net_result == expected_net
    assert classify_monetary_outcome(net_result) == expected_outcome


def test_split_push_does_not_break_existing_win_streak() -> None:
    tracker = MonetaryStreakTracker()
    tracker.observe(10.0)
    tracker.observe(box_round_net_result(settled_box_table([10.0, -10.0])))
    tracker.observe(10.0)

    assert tracker.summary()["win_streaks"]["frequency"] == {2: 1}


def test_seed_boundaries_finalize_separate_streaks() -> None:
    first_seed = MonetaryStreakTracker()
    first_seed.observe(10.0)
    second_seed = MonetaryStreakTracker()
    second_seed.observe(10.0)

    aggregate = aggregate_streak_metrics([first_seed.summary(), second_seed.summary()])

    assert aggregate["win_streaks"]["frequency"] == {1: 2}
    assert aggregate["win_streaks"]["streak_count"] == 2
    assert aggregate["win_streaks"]["represented_rounds"] == 2
