import pytest

from shufflemaster_sim.actions import ActionType
from shufflemaster_sim.card_sources import ScriptedCardSource
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.casino_blackjack import CasinoBlackjackGame
from shufflemaster_sim.hand_values import hand_value, is_natural_blackjack
from shufflemaster_sim.state import HandState


def card(rank: Rank, draw_id: int = 0) -> Card:
    return Card(
        rank=rank,
        suit="spades",
        physical_id=f"test:{rank}:{draw_id}",
        draw_id=draw_id,
    )


def make_dealt_table(
    player_cards: list[Rank],
    dealer_cards: list[Rank],
) -> tuple[CasinoBlackjackGame, object, object, HandState]:
    game = CasinoBlackjackGame()
    table = game.create_table(round_index=0)
    box = table.boxes[0]
    hand = box.hands[0]
    hand.cards = [card(rank, index) for index, rank in enumerate(player_cards)]
    table.dealer.cards = [
        card(rank, index + 100) for index, rank in enumerate(dealer_cards)
    ]
    return game, table, box, hand


def legal_for(
    player_cards: list[Rank], dealer_cards: list[Rank]
) -> frozenset[ActionType]:
    game, table, box, hand = make_dealt_table(player_cards, dealer_cards)
    return game.legal_actions(table=table, box=box, hand=hand)


def test_dealer_hits_soft_seventeen() -> None:
    game, table, _, _ = make_dealt_table(["T", "7"], ["A", "6"])
    source = ScriptedCardSource([("T", "hearts")])

    game.play_dealer(table, source)

    assert [drawn.rank for drawn in table.dealer.cards] == ["A", "6", "T"]


def test_dealer_stands_on_hard_seventeen() -> None:
    game, table, _, _ = make_dealt_table(["T", "7"], ["T", "7"])

    game.play_dealer(table, ScriptedCardSource([]))

    assert [drawn.rank for drawn in table.dealer.cards] == ["T", "7"]


def test_dealer_stands_on_soft_eighteen() -> None:
    game, table, _, _ = make_dealt_table(["T", "7"], ["A", "7"])

    game.play_dealer(table, ScriptedCardSource([]))

    assert [drawn.rank for drawn in table.dealer.cards] == ["A", "7"]


def test_dealer_stops_after_bust() -> None:
    game, table, _, _ = make_dealt_table(["T", "7"], ["T", "6"])
    source = ScriptedCardSource([("T", "hearts")])

    game.play_dealer(table, source)

    assert [drawn.rank for drawn in table.dealer.cards] == ["T", "6", "T"]


def test_dealer_does_not_draw_if_all_player_hands_busted() -> None:
    game, table, _, hand = make_dealt_table(["T", "6", "T"], ["6"])
    hand.outcome_label = "bust"
    hand.net_result = -10.0
    hand.is_terminal = True

    game.play_dealer(table, ScriptedCardSource([]))

    assert [drawn.rank for drawn in table.dealer.cards] == ["6"]


def test_dealer_ace_ten_is_blackjack() -> None:
    _, table, _, _ = make_dealt_table(["T", "7"], ["A", "T"])

    assert is_natural_blackjack(table.dealer.cards)


def test_dealer_ten_ace_is_blackjack() -> None:
    _, table, _, _ = make_dealt_table(["T", "7"], ["T", "A"])

    assert is_natural_blackjack(table.dealer.cards)


def test_dealer_ace_five_five_is_twenty_one_but_not_blackjack() -> None:
    _, table, _, _ = make_dealt_table(["T", "7"], ["A", "5", "5"])

    assert hand_value(table.dealer.cards).total == 21
    assert not is_natural_blackjack(table.dealer.cards)


def test_dealer_ten_five_six_is_twenty_one_but_not_blackjack() -> None:
    _, table, _, _ = make_dealt_table(["T", "7"], ["T", "5", "6"])

    assert hand_value(table.dealer.cards).total == 21
    assert not is_natural_blackjack(table.dealer.cards)


def test_hard_total_below_twelve_cannot_stand() -> None:
    actions = legal_for(["5", "6"], ["6"])

    assert ActionType.STAND not in actions
    assert ActionType.HIT in actions


@pytest.mark.parametrize("cards", [["4", "5"], ["4", "6"], ["5", "6"]])
def test_hard_nine_ten_eleven_can_double_on_first_two_cards(cards: list[Rank]) -> None:
    assert ActionType.DOUBLE in legal_for(cards, ["6"])


def test_hard_eight_cannot_double() -> None:
    assert ActionType.DOUBLE not in legal_for(["4", "4"], ["6"])


def test_ace_eight_counts_as_nine_for_double_eligibility() -> None:
    assert ActionType.DOUBLE in legal_for(["A", "8"], ["6"])


def test_ace_seven_counts_as_eight_and_cannot_double() -> None:
    assert ActionType.DOUBLE not in legal_for(["A", "7"], ["6"])


def test_ten_ace_blackjack_is_not_double_candidate() -> None:
    actions = legal_for(["T", "A"], ["6"])

    assert ActionType.DOUBLE not in actions


def test_split_allowed_for_pair_of_eights() -> None:
    assert ActionType.SPLIT in legal_for(["8", "8"], ["T"])


def test_split_allowed_for_ten_value_cards() -> None:
    assert ActionType.SPLIT in legal_for(["T", "Q"], ["6"])


def test_resplit_is_not_allowed_by_default() -> None:
    game, table, box, hand = make_dealt_table(["8", "8"], ["6"])
    source = ScriptedCardSource([("8", "hearts"), ("8", "clubs")])

    game.apply_action(
        table=table,
        box=box,
        hand=hand,
        action_type=ActionType.SPLIT,
        card_source=source,
    )

    assert ActionType.SPLIT not in game.legal_actions(
        table=table,
        box=box,
        hand=box.hands[0],
    )


def test_double_after_split_is_legal_when_split_hand_totals_eleven() -> None:
    game, table, box, hand = make_dealt_table(["5", "5"], ["6"])
    source = ScriptedCardSource([("6", "hearts"), ("2", "clubs")])

    game.apply_action(
        table=table,
        box=box,
        hand=hand,
        action_type=ActionType.SPLIT,
        card_source=source,
    )

    assert ActionType.DOUBLE in game.legal_actions(
        table=table,
        box=box,
        hand=box.hands[0],
    )


def test_split_aces_cannot_double_after_one_card() -> None:
    game, table, box, hand = make_dealt_table(["A", "A"], ["6"])
    source = ScriptedCardSource([("8", "hearts"), ("9", "clubs")])

    game.apply_action(
        table=table,
        box=box,
        hand=hand,
        action_type=ActionType.SPLIT,
        card_source=source,
    )

    assert ActionType.DOUBLE not in game.legal_actions(
        table=table,
        box=box,
        hand=box.hands[0],
    )
