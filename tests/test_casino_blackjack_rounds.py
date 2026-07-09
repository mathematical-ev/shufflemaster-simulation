# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

from collections.abc import Iterable

from shufflemaster_sim.actions import ActionType, GameAction
from shufflemaster_sim.card_sources import ScriptedCardSource
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.casino_blackjack import CasinoBlackjackGame
from shufflemaster_sim.hand_values import hand_value, is_natural_blackjack
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


def play_round(cards: list[tuple[Rank, str]], actions: list[ActionType]) -> TableState:
    source = ScriptedCardSource(cards)
    return CasinoBlackjackGame().play_round(
        round_index=0,
        card_source=source,
        strategy=QueueStrategy(actions),
    )


def test_player_blackjack_vs_dealer_non_blackjack_pays_three_to_two() -> None:
    table = play_round(
        [
            ("A", "spades"),
            ("9", "clubs"),
            ("K", "hearts"),
            ("7", "clubs"),
            ("T", "diamonds"),
        ],
        [],
    )

    assert table.boxes[0].hands[0].outcome_label == "blackjack_win"
    assert table.boxes[0].hands[0].net_result == 15.0
    assert [drawn.rank for drawn in table.dealer.cards] == ["9"]
    assert table.boxes[0].hands[0].is_collected
    assert [drawn.rank for drawn in table.discard_rack] == ["A", "K", "9"]


def test_player_blackjack_pushes_against_dealer_blackjack() -> None:
    table = play_round(
        [("A", "spades"), ("A", "clubs"), ("K", "hearts"), ("T", "diamonds")],
        [],
    )

    assert table.boxes[0].hands[0].outcome_label == "blackjack_push"
    assert table.boxes[0].hands[0].net_result == 0.0


def test_dealer_blackjack_beats_ordinary_player_hand() -> None:
    table = play_round(
        [("T", "spades"), ("A", "clubs"), ("9", "hearts"), ("T", "diamonds")],
        [ActionType.STAND],
    )

    hand = table.boxes[0].hands[0]
    assert hand.outcome_label == "dealer_blackjack_loss"
    assert hand.net_result == -10.0


def test_player_ordinary_win_pays_even_money() -> None:
    table = play_round(
        [("T", "spades"), ("9", "clubs"), ("9", "hearts"), ("8", "diamonds")],
        [ActionType.STAND],
    )

    assert table.boxes[0].hands[0].outcome_label == "win"
    assert table.boxes[0].hands[0].net_result == 10.0


def test_player_ordinary_loss_loses_wager() -> None:
    table = play_round(
        [("T", "spades"), ("9", "clubs"), ("7", "hearts"), ("T", "diamonds")],
        [ActionType.STAND],
    )

    assert table.boxes[0].hands[0].outcome_label == "loss"
    assert table.boxes[0].hands[0].net_result == -10.0


def test_ordinary_push_pays_zero() -> None:
    table = play_round(
        [("T", "spades"), ("9", "clubs"), ("8", "hearts"), ("9", "diamonds")],
        [ActionType.STAND],
    )

    assert table.boxes[0].hands[0].outcome_label == "push"
    assert table.boxes[0].hands[0].net_result == 0.0


def test_player_bust_loses_immediately() -> None:
    table = play_round(
        [("T", "spades"), ("6", "clubs"), ("6", "hearts"), ("T", "diamonds")],
        [ActionType.HIT],
    )

    hand = table.boxes[0].hands[0]
    assert hand.outcome_label == "bust"
    assert hand.net_result == -10.0
    assert [drawn.rank for drawn in table.dealer.cards] == ["6"]


def test_double_win_pays_double_wager() -> None:
    table = play_round(
        [
            ("5", "spades"),
            ("6", "clubs"),
            ("5", "hearts"),
            ("T", "diamonds"),
            ("T", "clubs"),
            ("9", "hearts"),
        ],
        [ActionType.DOUBLE],
    )

    hand = table.boxes[0].hands[0]
    assert hand.is_doubled
    assert hand.outcome_label == "win"
    assert hand.net_result == 20.0


def test_double_loss_loses_double_wager() -> None:
    table = play_round(
        [
            ("5", "spades"),
            ("9", "clubs"),
            ("5", "hearts"),
            ("2", "diamonds"),
            ("T", "clubs"),
        ],
        [ActionType.DOUBLE],
    )

    hand = table.boxes[0].hands[0]
    assert hand.is_doubled
    assert hand.outcome_label == "loss"
    assert hand.net_result == -20.0


def test_double_bust_loses_double_wager() -> None:
    game = CasinoBlackjackGame()
    table = game.create_table(round_index=0)
    hand = table.boxes[0].hands[0]
    hand.cards = [
        Card(rank="6", suit="spades", physical_id="test:0", draw_id=0),
        Card(rank="6", suit="hearts", physical_id="test:1", draw_id=1),
    ]
    hand.is_doubled = True

    game._hit(table, hand, ScriptedCardSource([("T", "clubs")]))

    assert hand.outcome_label == "bust"
    assert hand.net_result == -20.0
    assert hand.is_collected


def test_dealer_blackjack_after_double_loses_original_only() -> None:
    table = play_round(
        [
            ("5", "spades"),
            ("A", "clubs"),
            ("5", "hearts"),
            ("9", "diamonds"),
            ("T", "clubs"),
        ],
        [ActionType.DOUBLE],
    )

    hand = table.boxes[0].hands[0]
    assert hand.outcome_label == "dealer_blackjack_double_loss"
    assert hand.net_result == -10.0


def test_split_creates_two_hands() -> None:
    game = CasinoBlackjackGame()
    source = ScriptedCardSource(
        [
            ("8", "spades"),
            ("6", "clubs"),
            ("8", "hearts"),
            ("2", "diamonds"),
            ("3", "clubs"),
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

    assert len(box.hands) == 2
    assert all(hand.is_split_hand for hand in box.hands)


def test_split_aces_receive_one_card_each_and_stop() -> None:
    game = CasinoBlackjackGame()
    source = ScriptedCardSource(
        [
            ("A", "spades"),
            ("6", "clubs"),
            ("A", "hearts"),
            ("T", "diamonds"),
            ("9", "clubs"),
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

    assert len(box.hands) == 2
    assert all(hand.is_terminal for hand in box.hands)
    assert [len(hand.cards) for hand in box.hands] == [2, 2]


def test_split_ace_ten_is_twenty_one_but_not_blackjack() -> None:
    game = CasinoBlackjackGame()
    source = ScriptedCardSource(
        [
            ("A", "spades"),
            ("6", "clubs"),
            ("A", "hearts"),
            ("T", "diamonds"),
            ("9", "clubs"),
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

    split_ace_ten = box.hands[0]
    assert hand_value(split_ace_ten.cards).total == 21
    assert not is_natural_blackjack(
        split_ace_ten.cards,
        blackjack_eligible=split_ace_ten.blackjack_eligible,
    )


def test_split_ten_ace_is_twenty_one_but_not_blackjack() -> None:
    game = CasinoBlackjackGame()
    source = ScriptedCardSource(
        [
            ("T", "spades"),
            ("6", "clubs"),
            ("Q", "hearts"),
            ("A", "diamonds"),
            ("9", "clubs"),
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

    split_ten_ace = box.hands[0]
    assert hand_value(split_ten_ace.cards).total == 21
    assert not is_natural_blackjack(
        split_ten_ace.cards,
        blackjack_eligible=split_ten_ace.blackjack_eligible,
    )


def test_two_split_hand_wins_net_twenty() -> None:
    table = play_round(
        [
            ("8", "spades"),
            ("6", "clubs"),
            ("8", "hearts"),
            ("2", "diamonds"),
            ("9", "clubs"),
            ("9", "hearts"),
            ("Q", "diamonds"),
            ("T", "clubs"),
        ],
        [
            ActionType.SPLIT,
            ActionType.HIT,
            ActionType.STAND,
            ActionType.STAND,
        ],
    )

    assert sum(hand.net_result for hand in table.boxes[0].hands) == 20.0


def test_one_split_win_and_one_split_loss_net_zero() -> None:
    table = play_round(
        [
            ("8", "spades"),
            ("6", "clubs"),
            ("8", "hearts"),
            ("2", "diamonds"),
            ("9", "clubs"),
            ("9", "hearts"),
            ("Q", "diamonds"),
            ("2", "clubs"),
        ],
        [
            ActionType.SPLIT,
            ActionType.HIT,
            ActionType.STAND,
            ActionType.STAND,
        ],
    )

    assert [hand.net_result for hand in table.boxes[0].hands] == [10.0, -10.0]
    assert sum(hand.net_result for hand in table.boxes[0].hands) == 0.0


def test_two_split_hand_losses_net_minus_twenty() -> None:
    table = play_round(
        [
            ("8", "spades"),
            ("T", "clubs"),
            ("8", "hearts"),
            ("9", "diamonds"),
            ("8", "clubs"),
            ("Q", "diamonds"),
        ],
        [ActionType.SPLIT, ActionType.STAND, ActionType.STAND],
    )

    assert sum(hand.net_result for hand in table.boxes[0].hands) == -20.0


def test_doubled_split_hand_receives_one_card_and_settles_as_twenty_wager() -> None:
    table = play_round(
        [
            ("5", "spades"),
            ("6", "clubs"),
            ("5", "hearts"),
            ("6", "diamonds"),
            ("2", "clubs"),
            ("9", "hearts"),
            ("T", "diamonds"),
            ("9", "clubs"),
            ("T", "spades"),
        ],
        [ActionType.SPLIT, ActionType.DOUBLE, ActionType.HIT, ActionType.STAND],
    )

    doubled_hand = table.boxes[0].hands[0]
    assert doubled_hand.is_doubled
    assert len(doubled_hand.cards) == 3
    assert doubled_hand.net_result == 20.0


def test_dealer_blackjack_after_split_collects_original_only() -> None:
    table = play_round(
        [
            ("8", "spades"),
            ("A", "clubs"),
            ("8", "hearts"),
            ("2", "diamonds"),
            ("3", "clubs"),
            ("8", "diamonds"),
            ("7", "hearts"),
            ("T", "clubs"),
        ],
        [
            ActionType.SPLIT,
            ActionType.HIT,
            ActionType.STAND,
            ActionType.HIT,
            ActionType.STAND,
        ],
    )

    hands = table.boxes[0].hands
    assert sum(hand.net_result for hand in hands) == -10.0
    assert [hand.outcome_label for hand in hands] == [
        "dealer_blackjack_split_original_loss",
        "dealer_blackjack_split_standoff",
    ]


def test_dealer_blackjack_after_doubled_split_collects_original_only() -> None:
    table = play_round(
        [
            ("5", "spades"),
            ("A", "clubs"),
            ("5", "hearts"),
            ("6", "diamonds"),
            ("2", "clubs"),
            ("9", "hearts"),
            ("4", "diamonds"),
            ("T", "spades"),
            ("T", "clubs"),
        ],
        [ActionType.SPLIT, ActionType.DOUBLE, ActionType.HIT, ActionType.HIT],
    )

    hands = table.boxes[0].hands
    assert hands[0].is_doubled
    assert sum(hand.net_result for hand in hands) == -10.0


def test_dealer_blackjack_after_split_preserves_prior_bust_loss() -> None:
    table = play_round(
        [
            ("8", "spades"),
            ("A", "clubs"),
            ("8", "hearts"),
            ("8", "diamonds"),
            ("3", "clubs"),
            ("T", "hearts"),
            ("9", "diamonds"),
            ("T", "clubs"),
        ],
        [ActionType.SPLIT, ActionType.HIT, ActionType.HIT, ActionType.STAND],
    )

    hands = table.boxes[0].hands
    assert hands[0].outcome_label == "bust"
    assert hands[0].net_result == -10.0
    assert hands[1].outcome_label == "dealer_blackjack_split_standoff"
    assert sum(hand.net_result for hand in hands) == -10.0
