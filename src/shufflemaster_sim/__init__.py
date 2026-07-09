"""Foundational package for shufflemaster-sim."""

from shufflemaster_sim.card_sources import (
    CardSource,
    FiniteShoeCardSource,
    IidRandomCardSource,
    ManualShoeCardSource,
    One2SixCardSource,
    One2SixConfig,
    PhysicalIidCardSource,
    ScriptedCardSource,
)
from shufflemaster_sim.cards import RANKS, SUITS, Card, Rank, Suit, blackjack_value
from shufflemaster_sim.simulation import (
    SimulationConfig,
    run_casino_blackjack_baseline,
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
    "PhysicalIidCardSource",
    "Rank",
    "ScriptedCardSource",
    "SimulationConfig",
    "Suit",
    "blackjack_value",
    "run_casino_blackjack_baseline",
]
