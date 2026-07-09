# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Pure hand-value helpers for blackjack-like games."""

from dataclasses import dataclass

from shufflemaster_sim.cards import Card, blackjack_value


@dataclass(frozen=True, slots=True)
class HandValue:
    """Best blackjack hand total and whether an ace is counted as 11."""

    total: int
    is_soft: bool


def card_blackjack_value(card: Card) -> int:
    """Return the raw blackjack value for a single card."""
    return blackjack_value(card)


def hand_value(cards: list[Card]) -> HandValue:
    """Return the best blackjack total for a hand."""
    total = sum(card_blackjack_value(card) for card in cards)
    aces_counted_as_eleven = sum(1 for card in cards if card.rank == "A")

    while total > 21 and aces_counted_as_eleven > 0:
        total -= 10
        aces_counted_as_eleven -= 1

    return HandValue(total=total, is_soft=aces_counted_as_eleven > 0)


def is_soft_hand(cards: list[Card]) -> bool:
    """Return whether the best total counts an ace as 11."""
    return hand_value(cards).is_soft


def is_bust(cards: list[Card]) -> bool:
    """Return whether the hand exceeds 21."""
    return hand_value(cards).total > 21


def is_natural_blackjack(cards: list[Card], *, blackjack_eligible: bool = True) -> bool:
    """Return whether cards are an eligible two-card natural blackjack."""
    if not blackjack_eligible or len(cards) != 2:
        return False
    return hand_value(cards).total == 21 and any(card.rank == "A" for card in cards)


def split_value(card: Card) -> int | str:
    """Return the value used for pair splitting."""
    value = card_blackjack_value(card)
    if value == 11:
        return "A"
    if value == 10:
        return 10
    return value


def double_eligibility_total(cards: list[Card]) -> int:
    """Return the double-eligibility total with aces counted as 1."""
    total = 0
    for card in cards:
        total += 1 if card.rank == "A" else min(card_blackjack_value(card), 10)
    return total


def is_double_eligible_total(cards: list[Card]) -> bool:
    """Return whether the first two cards satisfy house double totals."""
    return len(cards) == 2 and double_eligibility_total(cards) in {9, 10, 11}
