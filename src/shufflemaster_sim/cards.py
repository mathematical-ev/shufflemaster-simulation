# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Basic card identity and raw blackjack card values."""

from dataclasses import dataclass
from typing import Final, Literal

Rank = Literal["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
Suit = Literal["clubs", "diamonds", "hearts", "spades"]

RANKS: Final[tuple[Rank, ...]] = (
    "A",
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
)
SUITS: Final[tuple[Suit, ...]] = ("clubs", "diamonds", "hearts", "spades")

_BLACKJACK_VALUES: Final[dict[Rank, int]] = {
    "A": 11,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
}


@dataclass(frozen=True, slots=True)
class Card:
    """A dealt card event.

    The physical id identifies the real card. The draw id identifies this
    specific draw event. Finite-shoe and future CSM sources preserve physical
    ids across discards and later redraws while assigning a fresh draw id.
    """

    rank: Rank
    suit: Suit
    physical_id: str
    draw_id: int


def blackjack_value(rank: Rank | Card) -> int:
    """Return the raw blackjack value for a rank or card."""
    if isinstance(rank, Card):
        rank = rank.rank
    return _BLACKJACK_VALUES[rank]
