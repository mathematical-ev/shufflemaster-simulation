# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

from collections.abc import Iterable, Sequence
from dataclasses import replace

from shufflemaster_sim.actions import ActionType, GameAction
from shufflemaster_sim.card_sources import CardSpec, FiniteShoeCardSource
from shufflemaster_sim.cards import Card
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.state import BoxState, HandState, TableState


class QueueStrategy:
    def __init__(self, actions: Iterable[ActionType]) -> None:
        self._actions = list(actions)

    def choose_action(
        self,
        *,
        table: TableState,
        box: BoxState,
        hand: HandState,
        dealer_upcard: Card,
        legal_actions: frozenset[ActionType],
    ) -> GameAction:
        _ = table, box, hand, dealer_upcard
        action = self._actions.pop(0)
        assert action in legal_actions
        return GameAction(action_type=action)


class StandOrHitStrategy:
    def choose_action(
        self,
        *,
        table: TableState,
        box: BoxState,
        hand: HandState,
        dealer_upcard: Card,
        legal_actions: frozenset[ActionType],
    ) -> GameAction:
        _ = table, box, hand, dealer_upcard
        if ActionType.STAND in legal_actions:
            return GameAction(action_type=ActionType.STAND)
        return GameAction(action_type=ActionType.HIT)


class TrackingCardSource:
    def __init__(self, cards: Iterable[CardSpec]) -> None:
        self._cards = iter(cards)
        self._next_draw_id = 0
        self.draw_count = 0
        self.accepted_discard_batches: list[list[Card]] = []
        self.accept_call_draw_counts: list[int] = []

    def before_round(self) -> None:
        pass

    def draw_card(self) -> Card:
        card_spec = next(self._cards)
        draw_id = self._next_draw_id
        self._next_draw_id += 1
        self.draw_count += 1
        if isinstance(card_spec, Card):
            return replace(card_spec, draw_id=draw_id)
        rank, suit = card_spec
        return Card(
            rank=rank,
            suit=suit,
            physical_id=f"tracking:{draw_id}",
            draw_id=draw_id,
        )

    def accept_discards(self, cards: Sequence[Card]) -> None:
        self.accepted_discard_batches.append(list(cards))
        self.accept_call_draw_counts.append(self.draw_count)


def play_round(
    game: CasinoBlackjackGame,
    source: TrackingCardSource,
    actions: list[ActionType],
    *,
    round_index: int = 0,
) -> TableState:
    return game.play_round(
        round_index=round_index,
        card_source=source,
        strategy=QueueStrategy(actions),
    )


def draw_ids(cards: Iterable[Card]) -> list[int]:
    return [card.draw_id for card in cards]


def test_bust_hand_cards_are_discarded_immediately_and_not_again() -> None:
    game = CasinoBlackjackGame()
    source = TrackingCardSource(
        [("T", "spades"), ("6", "clubs"), ("6", "hearts"), ("T", "diamonds")]
    )

    table = play_round(game, source, [ActionType.HIT])

    assert table.boxes[0].hands[0].is_collected
    assert draw_ids(table.discard_rack) == [0, 2, 3, 1]
    assert len(draw_ids(table.discard_rack)) == len(set(draw_ids(table.discard_rack)))


def test_live_player_hand_stays_on_layout_until_final_collection() -> None:
    game = CasinoBlackjackGame()
    source = TrackingCardSource(
        [
            ("T", "spades"),
            ("9", "clubs"),
            ("8", "hearts"),
            ("7", "diamonds"),
            ("T", "clubs"),
        ]
    )
    table = game.create_table(round_index=0)
    game.deal_initial_cards(table, source)
    hand = table.boxes[0].hands[0]

    game.apply_action(
        table=table,
        box=table.boxes[0],
        hand=hand,
        action_type=ActionType.STAND,
        card_source=source,
    )

    assert not hand.is_collected
    assert table.discard_rack == []

    game.play_dealer(table, source)
    game.settle(table)
    game.collect_remaining_layout_cards(table)

    assert hand.is_collected
    assert draw_ids(table.discard_rack) == [0, 2, 1, 3, 4]


def test_split_bust_discards_only_busted_hand_until_final_collection() -> None:
    game = CasinoBlackjackGame()
    source = TrackingCardSource(
        [
            ("8", "spades"),
            ("6", "clubs"),
            ("8", "hearts"),
            ("8", "diamonds"),
            ("3", "clubs"),
            ("T", "hearts"),
        ]
    )
    table = game.create_table(round_index=0)
    game.deal_initial_cards(table, source)
    box = table.boxes[0]
    game.apply_action(
        table=table,
        box=box,
        hand=box.hands[0],
        action_type=ActionType.SPLIT,
        card_source=source,
    )

    game.apply_action(
        table=table,
        box=box,
        hand=box.hands[0],
        action_type=ActionType.HIT,
        card_source=source,
    )

    assert box.hands[0].outcome_label == "bust"
    assert box.hands[0].is_collected
    assert not box.hands[1].is_collected
    assert draw_ids(table.discard_rack) == [0, 3, 5]


def test_previous_rack_returns_only_after_next_initial_deal() -> None:
    game = CasinoBlackjackGame()
    source = TrackingCardSource(
        [
            ("T", "spades"),
            ("6", "clubs"),
            ("6", "hearts"),
            ("T", "diamonds"),
            ("2", "spades"),
            ("3", "clubs"),
            ("4", "hearts"),
        ]
    )
    first_table = play_round(game, source, [ActionType.HIT], round_index=0)
    draw_count_after_first_round = source.draw_count

    assert source.accepted_discard_batches == []

    second_table = game.create_table(round_index=1)
    game.deal_initial_cards(second_table, source)

    assert source.accept_call_draw_counts == [draw_count_after_first_round + 3]
    assert draw_ids(source.accepted_discard_batches[0]) == draw_ids(
        first_table.discard_rack
    )
    assert second_table.discard_rack == []


def test_previous_rack_is_not_returned_when_shuffling_device_disabled() -> None:
    game = CasinoBlackjackGame(CasinoBlackjackConfig(use_shuffling_device=False))
    source = TrackingCardSource(
        [
            ("T", "spades"),
            ("6", "clubs"),
            ("6", "hearts"),
            ("T", "diamonds"),
            ("2", "spades"),
            ("3", "clubs"),
            ("4", "hearts"),
        ]
    )
    play_round(game, source, [ActionType.HIT], round_index=0)
    second_table = game.create_table(round_index=1)

    game.deal_initial_cards(second_table, source)

    assert source.accepted_discard_batches == []


def test_current_blackjack_discards_are_not_in_previous_rack_return() -> None:
    game = CasinoBlackjackGame()
    source = TrackingCardSource(
        [
            ("T", "spades"),
            ("6", "clubs"),
            ("6", "hearts"),
            ("T", "diamonds"),
            ("A", "spades"),
            ("6", "diamonds"),
            ("K", "hearts"),
        ]
    )
    first_table = play_round(game, source, [ActionType.HIT], round_index=0)
    second_table = game.create_table(round_index=1)

    game.deal_initial_cards(second_table, source)
    assert draw_ids(source.accepted_discard_batches[0]) == draw_ids(
        first_table.discard_rack
    )

    game.settle_immediate_blackjacks(second_table)

    assert draw_ids(source.accepted_discard_batches[0]) == draw_ids(
        first_table.discard_rack
    )
    assert draw_ids(second_table.discard_rack) == [4, 6]


def test_source_receives_discards_in_same_order_as_ordered_rack() -> None:
    game = CasinoBlackjackGame()
    source = TrackingCardSource(
        [
            ("T", "spades"),
            ("9", "clubs"),
            ("8", "hearts"),
            ("7", "diamonds"),
            ("T", "clubs"),
            ("2", "spades"),
            ("3", "clubs"),
            ("4", "hearts"),
        ]
    )
    first_table = play_round(game, source, [ActionType.STAND], round_index=0)
    second_table = game.create_table(round_index=1)

    game.deal_initial_cards(second_table, source)

    assert source.accepted_discard_batches[0] == first_table.discard_rack
    assert len(draw_ids(source.accepted_discard_batches[0])) == len(
        set(draw_ids(source.accepted_discard_batches[0]))
    )


def test_finite_shoe_receives_previous_rack_after_next_initial_deal() -> None:
    game = CasinoBlackjackGame()
    source = FiniteShoeCardSource(deck_count=1, seed=42)
    first_table = game.play_round(
        round_index=0,
        card_source=source,
        strategy=StandOrHitStrategy(),
    )

    assert source.accepted_discard_batches == []

    second_table = game.create_table(round_index=1)
    game.deal_initial_cards(second_table, source)

    assert source.accepted_discard_batches == [first_table.discard_rack]
    assert source.discard_tray == first_table.discard_rack
    assert second_table.discard_rack == []
