import pytest

from shufflemaster_sim.actions import ActionType
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.star_blackjack import StarBlackjackGame
from shufflemaster_sim.state import HandState
from shufflemaster_sim.strategies.published_star_strategy import (
    PublishedApproxStarStrategy,
)


def card(rank: Rank, draw_id: int = 0) -> Card:
    return Card(
        rank=rank,
        suit="spades",
        physical_id=f"test:{rank}:{draw_id}",
        draw_id=draw_id,
    )


def choose_action(player_cards: list[Rank], dealer_rank: Rank) -> ActionType:
    game = StarBlackjackGame()
    table = game.create_table(round_index=0)
    box = table.boxes[0]
    hand = box.hands[0]
    hand.cards = [card(rank, index) for index, rank in enumerate(player_cards)]
    dealer_upcard = card(dealer_rank, 100)
    table.dealer.cards = [dealer_upcard]
    legal_actions = game.legal_actions(table=table, box=box, hand=hand)

    return (
        PublishedApproxStarStrategy()
        .choose_action(
            table=table,
            box=box,
            hand=hand,
            dealer_upcard=dealer_upcard,
            legal_actions=legal_actions,
        )
        .action_type
    )


def test_hard_sixteen_versus_dealer_ten_hits() -> None:
    assert choose_action(["T", "6"], "T") == ActionType.HIT


def test_hard_sixteen_versus_dealer_six_stands() -> None:
    assert choose_action(["T", "6"], "6") == ActionType.STAND


def test_hard_eleven_versus_dealer_ace_doubles_if_legal() -> None:
    assert choose_action(["5", "6"], "A") == ActionType.DOUBLE


def test_soft_eighteen_versus_dealer_nine_hits() -> None:
    assert choose_action(["A", "7"], "9") == ActionType.HIT


def test_soft_nineteen_versus_dealer_six_doubles_if_legal() -> None:
    assert choose_action(["A", "8"], "6") == ActionType.DOUBLE


def test_soft_eighteen_versus_dealer_two_falls_back_to_stand() -> None:
    assert choose_action(["A", "7"], "2") == ActionType.STAND


def test_pair_of_eights_versus_ten_splits_if_legal() -> None:
    assert choose_action(["8", "8"], "T") == ActionType.SPLIT


def test_pair_of_tens_versus_six_stands() -> None:
    assert choose_action(["T", "T"], "6") == ActionType.STAND


def test_pair_of_aces_versus_ace_splits_if_legal() -> None:
    assert choose_action(["A", "A"], "A") == ActionType.SPLIT


@pytest.mark.parametrize(
    ("player_cards", "dealer_rank"),
    [
        (["T", "6"], "T"),
        (["T", "6"], "6"),
        (["5", "6"], "A"),
        (["A", "7"], "2"),
        (["A", "8"], "6"),
        (["8", "8"], "T"),
        (["T", "T"], "6"),
        (["A", "A"], "A"),
    ],
)
def test_strategy_never_chooses_illegal_action(
    player_cards: list[Rank],
    dealer_rank: Rank,
) -> None:
    game = StarBlackjackGame()
    table = game.create_table(round_index=0)
    box = table.boxes[0]
    hand: HandState = box.hands[0]
    hand.cards = [card(rank, index) for index, rank in enumerate(player_cards)]
    dealer_upcard = card(dealer_rank, 100)
    table.dealer.cards = [dealer_upcard]
    legal_actions = game.legal_actions(table=table, box=box, hand=hand)

    action = PublishedApproxStarStrategy().choose_action(
        table=table,
        box=box,
        hand=hand,
        dealer_upcard=dealer_upcard,
        legal_actions=legal_actions,
    )

    assert action.action_type in legal_actions


def test_strategy_declines_insurance_and_even_money() -> None:
    strategy = PublishedApproxStarStrategy()

    assert not strategy.wants_insurance()
    assert not strategy.wants_even_money()


def test_strategy_uses_double_fallback_when_star_double_is_illegal() -> None:
    assert choose_action(["A", "7"], "2") == ActionType.STAND


def test_strategy_uses_pair_fallback_when_resplit_is_illegal() -> None:
    game = StarBlackjackGame()
    table = game.create_table(round_index=0)
    box = table.boxes[0]
    hand = box.hands[0]
    hand.cards = [card("8", 0), card("8", 1)]
    hand.is_split_hand = True
    dealer_upcard = card("T", 100)
    table.dealer.cards = [dealer_upcard]
    legal_actions = game.legal_actions(table=table, box=box, hand=hand)

    action = PublishedApproxStarStrategy().choose_action(
        table=table,
        box=box,
        hand=hand,
        dealer_upcard=dealer_upcard,
        legal_actions=legal_actions,
    )

    assert ActionType.SPLIT not in legal_actions
    assert action.action_type == ActionType.HIT
