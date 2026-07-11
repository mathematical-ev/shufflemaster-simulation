# SPDX-License-Identifier: GPL-3.0-or-later

import csv
from collections.abc import Iterable
from pathlib import Path

import pytest
from experiments.multi_box_counterfactual import (
    COUNTERFACTUAL_ACTION_FIELDS,
    CounterfactualSnapshot,
    MultiBoxCounterfactualConfig,
    SampledState,
    _aggregate_outcome,
    _make_source,
    branch_from_snapshot,
    clone_snapshot,
    observable_rack_state,
    paired_marginal_records,
    run_multi_box_counterfactual_experiment,
    state_bucket,
)

from shufflemaster_sim.actions import ActionType, GameAction
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
from shufflemaster_sim.state import BlackjackDecisionState
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)


class RecordingStandStrategy:
    def __init__(self) -> None:
        self.player_ranks: list[tuple[Rank, ...]] = []

    def choose_action(
        self,
        *,
        decision: BlackjackDecisionState,
    ) -> GameAction:
        self.player_ranks.append(decision.player_ranks)
        action = (
            ActionType.STAND
            if ActionType.STAND in decision.legal_actions
            else ActionType.HIT
        )
        return GameAction(action_type=action)


class QueueStrategy:
    def __init__(self, actions: Iterable[ActionType]) -> None:
        self.actions = list(actions)

    def choose_action(
        self,
        *,
        decision: BlackjackDecisionState,
    ) -> GameAction:
        action = self.actions.pop(0)
        assert action in decision.legal_actions
        return GameAction(action_type=action)


class DrawCountStandStrategy:
    def __init__(self, source: ScriptedCardSource) -> None:
        self.source = source
        self.draw_counts: list[int] = []

    def choose_action(
        self,
        *,
        decision: BlackjackDecisionState,
    ) -> GameAction:
        self.draw_counts.append(self.source._next_draw_id)
        assert ActionType.STAND in decision.legal_actions
        return GameAction(action_type=ActionType.STAND)


def cards_for_initial_deal(box_count: int) -> list[Card]:
    ranks: tuple[Rank, ...] = (
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "T",
        "J",
        "Q",
        "K",
        "A",
        "2",
        "3",
    )
    return [
        Card(
            rank=ranks[index],
            suit="spades",
            physical_id=f"initial:{index}",
            draw_id=-1,
        )
        for index in range(2 * box_count + 1)
    ]


@pytest.mark.parametrize("box_count", [1, 2, 4, 7])
def test_exact_multi_box_initial_deal_order(box_count: int) -> None:
    source = ScriptedCardSource(cards_for_initial_deal(box_count))
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(box_count=box_count, burn_initial_card=False)
    )
    table = game.create_table(round_index=0)

    game.deal_initial_cards(table, source)

    assert table.dealer.cards[0].draw_id == box_count
    for box_index, box in enumerate(table.boxes):
        assert [card.draw_id for card in box.hands[0].cards] == [
            box_index,
            box_count + 1 + box_index,
        ]


def test_boxes_act_in_table_order_with_same_strategy() -> None:
    source = ScriptedCardSource(
        [
            ("2", "spades"),
            ("3", "spades"),
            ("4", "spades"),
            ("5", "spades"),
            ("6", "clubs"),
            ("T", "hearts"),
            ("T", "diamonds"),
            ("T", "clubs"),
            ("T", "spades"),
        ]
    )
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(box_count=4, burn_initial_card=False)
    )
    table = game.create_table(round_index=0)
    game.deal_initial_cards(table, source)
    strategy = RecordingStandStrategy()

    game.play_player_hands(table, source, strategy)

    assert [ranks[0] for ranks in strategy.player_ranks] == ["2", "3", "4", "5"]


def test_all_boxes_finish_actions_before_dealer_completion() -> None:
    source = ScriptedCardSource(
        [
            ("T", "spades"),
            ("9", "clubs"),
            ("6", "hearts"),
            ("7", "diamonds"),
            ("8", "spades"),
            ("T", "clubs"),
            ("5", "hearts"),
        ]
    )
    strategy = DrawCountStandStrategy(source)
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(box_count=2, burn_initial_card=False)
    )

    table = game.play_round(round_index=0, card_source=source, strategy=strategy)

    assert strategy.draw_counts == [5, 5]
    assert [card.draw_id for card in table.dealer.cards] == [2, 5, 6]


def test_multi_box_double_split_and_net_are_attributed_by_box() -> None:
    source = ScriptedCardSource(
        [
            ("5", "spades"),
            ("8", "clubs"),
            ("6", "hearts"),
            ("5", "diamonds"),
            ("8", "spades"),
            ("T", "clubs"),
            ("T", "hearts"),
            ("9", "diamonds"),
            ("T", "spades"),
            ("5", "clubs"),
        ]
    )
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(box_count=2, burn_initial_card=False)
    )
    table = game.play_round(
        round_index=0,
        card_source=source,
        strategy=QueueStrategy(
            [
                ActionType.DOUBLE,
                ActionType.SPLIT,
                ActionType.STAND,
                ActionType.STAND,
            ]
        ),
    )

    assert sum(hand.is_doubled for hand in table.boxes[0].hands) == 1
    assert len(table.boxes[0].hands) == 1
    assert len(table.boxes[1].hands) == 2
    assert sum(hand.is_doubled for hand in table.boxes[1].hands) == 0
    assert sum(hand.net_result for box in table.boxes for hand in box.hands) == sum(
        sum(hand.net_result for hand in box.hands) for box in table.boxes
    )


@pytest.mark.parametrize(
    ("box_results", "expected"),
    [([10.0, -10.0], "push"), ([20.0, -10.0], "win"), ([10.0, -20.0], "loss")],
)
def test_aggregate_outcome_uses_all_box_results(
    box_results: list[float], expected: str
) -> None:
    assert _aggregate_outcome(sum(box_results)) == expected


def test_previous_rack_returns_after_complete_seven_box_initial_deal() -> None:
    source = One2SixCardSource(config=One2SixConfig(deck_count=6), seed=42)
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            box_count=7,
            burn_initial_card=False,
            use_shuffling_device=True,
        )
    )
    first_table = game.play_round(
        round_index=0,
        card_source=source,
        strategy=PublishedApproxCasinoStrategy(),
    )
    before_second_deal = source.draw_count
    second_table = game.create_table(round_index=1)

    game.deal_initial_cards(second_table, source)

    assert source.draw_count - before_second_deal == 15
    assert source.accepted_discard_batches[-1] == first_table.discard_rack
    assert game.pending_discard_rack == ()


def test_observable_rack_features_and_bucket_boundaries() -> None:
    cards = [
        make_card("2", 0),
        make_card("6", 1),
        make_card("8", 2),
        make_card("T", 3),
        make_card("A", 4),
    ]

    observable = observable_rack_state(cards)

    assert observable.rack_size == 5
    assert observable.rack_low_count == 2
    assert observable.rack_neutral_count == 1
    assert observable.rack_ten_value_count == 1
    assert observable.rack_ace_count == 1
    assert observable.rack_hi_lo_count == 0
    assert observable.normalized_exclusion_score == 0.0
    assert [state_bucket(value) for value in (-6, -5, -3, -2, 2, 3, 5, 6)] == [
        "strong_high_heavy",
        "moderate_high_heavy",
        "moderate_high_heavy",
        "neutral",
        "neutral",
        "moderate_low_heavy",
        "moderate_low_heavy",
        "strong_low_heavy",
    ]


def test_observable_export_contains_no_hidden_state() -> None:
    record = observable_rack_state([make_card("5", 0)]).as_feature_record()

    assert set(record) == {
        "rack_size",
        "rack_hi_lo_count",
        "rack_low_count",
        "rack_neutral_count",
        "rack_ten_value_count",
        "rack_ace_count",
        "normalized_exclusion_score",
        "state_bucket",
    }
    assert not any(
        hidden in field
        for field in record
        for hidden in ("physical", "rng", "feeder", "carousel", "buffer", "shelf")
    )


def test_snapshot_clones_rng_and_mutable_source_state() -> None:
    snapshot, sampled_state = prepared_snapshot()
    first = clone_snapshot(snapshot)
    second = clone_snapshot(snapshot)

    assert first.card_source._rng.getstate() == second.card_source._rng.getstate()
    assert first.card_source._rng is not second.card_source._rng
    assert first.card_source._carousel is not second.card_source._carousel
    assert (
        first.card_source.output_buffer_cards == second.card_source.output_buffer_cards
    )

    first_result = branch_from_snapshot(
        first,
        sampled_state=sampled_state,
        box_count=3,
        base_bet=10.0,
        strategy=PublishedApproxCasinoStrategy(),
    )

    assert first_result.box_count == 3
    assert second.card_source.draw_count == snapshot.card_source.draw_count
    assert second.game.pending_discard_rack == snapshot.game.pending_discard_rack


def test_repeating_same_snapshot_branch_is_deterministic_and_isolated() -> None:
    snapshot, sampled_state = prepared_snapshot()
    before_draw_count = snapshot.card_source.draw_count
    before_rack = snapshot.game.pending_discard_rack

    first = branch_from_snapshot(
        snapshot,
        sampled_state=sampled_state,
        box_count=4,
        base_bet=10.0,
        strategy=PublishedApproxCasinoStrategy(),
    )
    second = branch_from_snapshot(
        snapshot,
        sampled_state=sampled_state,
        box_count=4,
        base_bet=10.0,
        strategy=PublishedApproxCasinoStrategy(),
    )

    assert first == second
    assert snapshot.card_source.draw_count == before_draw_count
    assert snapshot.game.pending_discard_rack == before_rack


def test_default_counterfactual_creates_exactly_seven_branches() -> None:
    snapshot, sampled_state = prepared_snapshot()

    branches = [
        branch_from_snapshot(
            snapshot,
            sampled_state=sampled_state,
            box_count=box_count,
            base_bet=10.0,
            strategy=PublishedApproxCasinoStrategy(),
        )
        for box_count in range(1, 8)
    ]

    assert [branch.box_count for branch in branches] == list(range(1, 8))
    assert [len(branch.box_positions) for branch in branches] == list(range(1, 8))
    assert [branch.initial_wager for branch in branches] == [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
    ]


def test_paired_marginals_require_matching_state_branches() -> None:
    snapshot, sampled_state = prepared_snapshot()
    branches = [
        branch_from_snapshot(
            snapshot,
            sampled_state=sampled_state,
            box_count=box_count,
            base_bet=10.0,
            strategy=PublishedApproxCasinoStrategy(),
        )
        for box_count in (1, 2, 3)
    ]

    marginals = paired_marginal_records(branches, added_box_numbers=(2, 3))
    assert [item.state_id for item in marginals] == [
        sampled_state.state_id,
        sampled_state.state_id,
    ]
    assert marginals[0].marginal_net == (
        branches[1].total_player_net - branches[0].total_player_net
    )
    with pytest.raises(RuntimeError, match="Missing paired branches"):
        paired_marginal_records(branches[:2], added_box_numbers=(3,))


def test_small_experiment_reconciles_and_writes_only_compact_actions(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "counterfactual"
    config = MultiBoxCounterfactualConfig(
        states_per_seed=3,
        seeds=(42,),
        burn_in_rounds=1,
        sample_interval_rounds=1,
        output_dir=output_dir,
    )

    summary = run_multi_box_counterfactual_experiment(config)

    expected_files = {
        "summary.json",
        "summary.md",
        "state_frequency.csv",
        "state_action_values.csv",
        "per_seed_state_action_values.csv",
        "marginal_box_values.csv",
        "box_position_summary.csv",
        "counterfactual_actions.csv",
        "state_action_edge_heatmap_physical_iid.png",
        "state_action_edge_heatmap_one2six.png",
        "marginal_box_value_heatmap_physical_iid.png",
        "marginal_box_value_heatmap_one2six.png",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    with (output_dir / "counterfactual_actions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 1 * 3 * 7
    assert tuple(rows[0]) == COUNTERFACTUAL_ACTION_FIELDS
    assert summary["hidden_state_exported"] is False
    assert (
        sum(
            row["state_count"]
            for row in summary["state_frequency"]
            if row["seed_or_aggregate"] == "aggregate"
        )
        == 6
    )


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="states_per_seed"):
        MultiBoxCounterfactualConfig(states_per_seed=0)
    with pytest.raises(ValueError, match="six decks"):
        MultiBoxCounterfactualConfig(deck_count=8)
    with pytest.raises(ValueError, match="unique"):
        MultiBoxCounterfactualConfig(box_counts=(1, 1))
    with pytest.raises(ValueError, match="1 through 7"):
        MultiBoxCounterfactualConfig(box_counts=(0, 1))


def make_card(rank: Rank, index: int) -> Card:
    return Card(
        rank=rank,
        suit="spades",
        physical_id=f"observable:{index}",
        draw_id=index,
    )


def prepared_snapshot() -> tuple[CounterfactualSnapshot, SampledState]:
    source = _make_source("one2six", 6, 42)
    game = CasinoBlackjackGame(CasinoBlackjackConfig(base_bet=10.0, box_count=1))
    game.play_round(
        round_index=0,
        card_source=source,
        strategy=PublishedApproxCasinoStrategy(),
    )
    observable = observable_rack_state(game.pending_discard_rack)
    sampled_state = SampledState(
        source="one2six",
        seed=42,
        state_id="one2six:42:test",
        observable=observable,
    )
    return (
        CounterfactualSnapshot(
            game=game,
            card_source=source,
            next_round_index=1,
        ),
        sampled_state,
    )
