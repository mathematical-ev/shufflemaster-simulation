"""Simulation runners."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from shufflemaster_sim.card_sources import (
    CardSource,
    FiniteShoeCardSource,
    IidRandomCardSource,
    ManualShoeCardSource,
)
from shufflemaster_sim.games.star_blackjack import (
    StarBlackjackConfig,
    StarBlackjackGame,
)
from shufflemaster_sim.results import ResultRecorder, SimulationResult
from shufflemaster_sim.strategies.published_star_strategy import (
    PublishedApproxStarStrategy,
)

CardSourceKind = Literal["iid", "finite-shoe", "manual-shoe"]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for the Star Blackjack baseline simulation."""

    rounds: int = 10_000
    base_bet: float = 10.0
    box_count: int = 1
    box_bets: Mapping[int, float] | None = None
    seed: int | None = None
    card_source: CardSourceKind = "iid"
    deck_count: int | None = None
    cut_card_penetration: float = 0.75

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        box_bets = self.resolved_box_bets()
        if set(box_bets) != {1}:
            raise ValueError("Only box 1 is supported in runnable simulations for now.")
        if self.box_count != 1:
            raise ValueError("Only one-box baseline simulations are supported for now.")
        if self.card_source not in {"iid", "finite-shoe", "manual-shoe"}:
            raise ValueError(
                "card_source must be 'iid', 'finite-shoe', or 'manual-shoe'."
            )
        if not 0 < self.cut_card_penetration <= 1:
            raise ValueError("cut_card_penetration must be in (0, 1].")

    @property
    def effective_deck_count(self) -> int:
        """Return source-specific default deck count when omitted."""
        if self.deck_count is not None:
            return self.deck_count
        if self.card_source == "manual-shoe":
            return 8
        return 6

    def resolved_box_bets(self) -> dict[int, float]:
        """Return independent box wagers for the configured simulation."""
        if self.box_bets is None:
            return {1: self.base_bet}
        box_bets = dict(self.box_bets)
        if any(bet <= 0 for bet in box_bets.values()):
            raise ValueError("all box bets must be positive.")
        return box_bets


def run_star_blackjack_baseline(config: SimulationConfig) -> SimulationResult:
    """Run the one-box Star Blackjack baseline simulation."""
    card_source: CardSource
    if config.card_source == "iid":
        card_source = IidRandomCardSource(seed=config.seed)
    elif config.card_source == "finite-shoe":
        card_source = FiniteShoeCardSource(
            deck_count=config.effective_deck_count,
            seed=config.seed,
        )
    else:
        card_source = ManualShoeCardSource(
            deck_count=config.effective_deck_count,
            cut_card_penetration=config.cut_card_penetration,
            seed=config.seed,
        )

    box_bets = config.resolved_box_bets()
    game = StarBlackjackGame(
        StarBlackjackConfig(
            base_bet=box_bets[1],
            box_count=len(box_bets),
            box_bets=box_bets,
            deck_count=config.effective_deck_count,
        )
    )
    strategy = PublishedApproxStarStrategy()
    recorder = ResultRecorder(base_bet=box_bets[1], box_count=len(box_bets))

    for round_index in range(config.rounds):
        table = game.play_round(
            round_index=round_index,
            card_source=card_source,
            strategy=strategy,
        )
        recorder.record_round(table)

    result = recorder.build_result()
    if isinstance(card_source, ManualShoeCardSource):
        result.shuffle_count = card_source.shuffle_count
    return result
