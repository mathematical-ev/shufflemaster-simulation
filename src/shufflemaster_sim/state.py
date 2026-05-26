"""Mutable round state for blackjack-like table games."""

from dataclasses import dataclass, field

from shufflemaster_sim.cards import Card


@dataclass(slots=True)
class HandState:
    """State for one playable or settled hand."""

    hand_id: int
    cards: list[Card]
    wager: float
    is_split_hand: bool = False
    blackjack_eligible: bool = True
    is_doubled: bool = False
    is_from_split_aces: bool = False
    is_terminal: bool = False
    is_collected: bool = False
    outcome_label: str | None = None
    net_result: float = 0.0


@dataclass(slots=True)
class BoxState:
    """State for one player box."""

    box_id: int
    base_bet: float
    hands: list[HandState] = field(default_factory=list)


@dataclass(slots=True)
class DealerState:
    """State for the dealer hand."""

    cards: list[Card] = field(default_factory=list)
    is_collected: bool = False


@dataclass(slots=True)
class TableState:
    """State for a single table round."""

    boxes: list[BoxState]
    dealer: DealerState
    round_index: int
    discard_rack: list[Card] = field(default_factory=list)
