from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.hand_values import (
    hand_value,
    is_bust,
    is_natural_blackjack,
    split_value,
)


def card(rank: Rank, draw_id: int = 0) -> Card:
    return Card(
        rank=rank,
        suit="spades",
        physical_id=f"test:{rank}:{draw_id}",
        draw_id=draw_id,
    )


def test_ace_six_is_soft_seventeen() -> None:
    assert hand_value([card("A"), card("6")]).total == 17
    assert hand_value([card("A"), card("6")]).is_soft


def test_ace_six_ten_is_hard_seventeen() -> None:
    value = hand_value([card("A"), card("6"), card("T")])

    assert value.total == 17
    assert not value.is_soft


def test_pair_of_aces_is_soft_twelve() -> None:
    value = hand_value([card("A"), card("A")])

    assert value.total == 12
    assert value.is_soft


def test_two_aces_and_nine_is_soft_twenty_one() -> None:
    value = hand_value([card("A"), card("A"), card("9")])

    assert value.total == 21
    assert value.is_soft


def test_ten_seven_is_hard_seventeen() -> None:
    value = hand_value([card("T"), card("7")])

    assert value.total == 17
    assert not value.is_soft


def test_ten_six_seven_busts() -> None:
    cards = [card("T"), card("6"), card("7")]

    assert hand_value(cards).total == 23
    assert is_bust(cards)


def test_natural_blackjack_detection_requires_eligible_two_card_hand() -> None:
    natural = [card("A"), card("K")]

    assert is_natural_blackjack(natural)
    assert not is_natural_blackjack(natural, blackjack_eligible=False)
    assert not is_natural_blackjack([card("A"), card("9"), card("A")])


def test_split_ace_ten_is_twenty_one_but_not_blackjack() -> None:
    split_hand = [card("A"), card("T")]

    assert hand_value(split_hand).total == 21
    assert not is_natural_blackjack(split_hand, blackjack_eligible=False)


def test_ten_value_cards_share_split_value() -> None:
    assert split_value(card("T")) == split_value(card("Q")) == 10
