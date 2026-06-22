"""Foundational package for shufflemaster-sim."""

from shufflemaster_sim.card_sources import (
    CardSource,
    FiniteShoeCardSource,
    IidRandomCardSource,
    ManualShoeCardSource,
    One2SixCardSource,
    One2SixConfig,
    ScriptedCardSource,
)
from shufflemaster_sim.cards import RANKS, SUITS, Card, Rank, Suit, blackjack_value
from shufflemaster_sim.simulation import (
    SimulationConfig,
    run_star_blackjack_baseline,
)

__all__ = [
    "RANKS",
    "SUITS",
    "Card",
    "CardSource",
    "FiniteShoeCardSource",
    "IidRandomCardSource",
    "ManualShoeCardSource",
    "One2SixCardSource",
    "One2SixConfig",
    "Rank",
    "ScriptedCardSource",
    "SimulationConfig",
    "Suit",
    "blackjack_value",
    "run_star_blackjack_baseline",
]
