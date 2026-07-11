# SPDX-License-Identifier: GPL-3.0-or-later

"""Held-out conditional profitability for the frozen fading-exclusion score."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from random import Random
from typing import Any, Final, Literal

from experiments.fading_exclusion_validation import (
    FROZEN_WEIGHTS,
    CohortCounts,
    FadingState,
    LedgerCardSource,
    ObservableCohortLedger,
    RegressionDiagnostics,
    RunningStats,
    _play_round_capture_initial,
    calculate_fading_state,
)
from experiments.multi_box_counterfactual import SourceName, _make_source
from experiments.observable_card_response import CardIndicators, card_indicators
from experiments.plots import (
    plot_actual_vs_placebo_slopes,
    plot_conditional_edge_by_band,
    plot_initial_composition_by_band,
    plot_monetary_slopes_by_seed,
    plot_score_band_frequency,
)
from experiments.single_box_game_validation import student_t_summary
from shufflemaster_sim.actions import GameAction
from shufflemaster_sim.card_sources import One2SixCardSource
from shufflemaster_sim.cards import Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
    CasinoBlackjackStrategy,
)
from shufflemaster_sim.hand_values import is_natural_blackjack, split_value_from_rank
from shufflemaster_sim.state import BlackjackDecisionState, TableState
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)

Source = Literal["physical_iid", "one2six"]
ScoreBand = Literal[
    "strong_high_rich",
    "moderate_high_rich",
    "neutral",
    "moderate_low_rich",
    "strong_low_rich",
]

SOURCE_NAMES: Final[tuple[Source, ...]] = ("physical_iid", "one2six")
DEFAULT_DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = tuple(range(42, 52))
DEFAULT_VALIDATION_SEEDS: Final[tuple[int, ...]] = tuple(range(52, 62))
SCORE_BANDS: Final[tuple[ScoreBand, ...]] = (
    "strong_high_rich",
    "moderate_high_rich",
    "neutral",
    "moderate_low_rich",
    "strong_low_rich",
)
BAND_SCORES: Final[dict[ScoreBand, int]] = {
    "strong_high_rich": -2,
    "moderate_high_rich": -1,
    "neutral": 0,
    "moderate_low_rich": 1,
    "strong_low_rich": 2,
}
CONTRASTS: Final[dict[str, tuple[ScoreBand, ScoreBand]]] = {
    "strong_high_rich_minus_neutral": ("strong_high_rich", "neutral"),
    "moderate_high_rich_minus_neutral": ("moderate_high_rich", "neutral"),
    "strong_high_rich_minus_strong_low_rich": (
        "strong_high_rich",
        "strong_low_rich",
    ),
}
PERMUTATION_SEED_OFFSET: Final[int] = 8_671_309
PRIVATE_TERMS: Final[tuple[str, ...]] = (
    "physical_id",
    "draw_id",
    "shelf_id",
    "buffer_contents",
    "feeder_contents",
    "carousel_contents",
    "rng_state",
    "source_snapshot",
    "hidden_telemetry",
)

MONETARY_FIELDS: Final[tuple[str, ...]] = (
    "player_edge_per_initial_wager",
    "player_edge_per_total_wager",
    "mean_net_per_round",
    "round_net_sample_standard_deviation",
    "naive_round_standard_error",
)
EVENT_FIELDS: Final[tuple[str, ...]] = (
    "player_blackjack_rate",
    "double_action_rate",
    "split_action_rate",
    "winning_round_rate",
    "losing_round_rate",
    "push_round_rate",
    "average_cards_consumed",
)
COMPOSITION_FIELDS: Final[tuple[str, ...]] = (
    "player_first_hi_lo_mean",
    "dealer_upcard_hi_lo_mean",
    "player_second_hi_lo_mean",
    "combined_initial_hi_lo_mean",
    "player_card_low_rate",
    "player_card_ten_value_rate",
    "player_card_ace_rate",
    "dealer_upcard_low_rate",
    "dealer_upcard_ten_value_rate",
    "dealer_upcard_ace_rate",
)


@dataclass(frozen=True, slots=True)
class ConditionalProfitabilityConfig:
    """Configuration for development cutpoints and held-out monetary validation."""

    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS
    validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS
    development_rounds_per_seed: int = 20_000
    validation_rounds_per_seed: int = 100_000
    burn_in_rounds: int = 1_000
    deck_count: int = 6
    base_bet: float = 10.0
    score_quantiles: tuple[float, ...] = (0.10, 0.30, 0.70, 0.90)
    current_rack_weight: float = 1.00
    returned_1_15_weight: float = 0.75
    returned_16_50_weight: float = 0.40
    returned_51_100_weight: float = 0.20
    returned_over_100_weight: float = 0.00
    output_dir: Path = Path("experiments/outputs/conditional_profitability_validation")

    def __post_init__(self) -> None:
        if not self.development_seeds or not self.validation_seeds:
            raise ValueError("development and validation seeds must be nonempty.")
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("development seeds must be unique.")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation seeds must be unique.")
        if set(self.development_seeds).intersection(self.validation_seeds):
            raise ValueError("development and validation seeds must be disjoint.")
        if self.deck_count != 6:
            raise ValueError("conditional profitability requires exactly six decks.")
        if self.development_rounds_per_seed <= 0:
            raise ValueError("development_rounds_per_seed must be positive.")
        if self.validation_rounds_per_seed <= 0:
            raise ValueError("validation_rounds_per_seed must be positive.")
        if self.burn_in_rounds < 0:
            raise ValueError("burn_in_rounds must be non-negative.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        if len(self.score_quantiles) != 4:
            raise ValueError("exactly four score quantiles are required.")
        if tuple(sorted(self.score_quantiles)) != self.score_quantiles:
            raise ValueError("score quantiles must be ordered.")
        if len(set(self.score_quantiles)) != len(self.score_quantiles) or any(
            not 0 < quantile < 1 for quantile in self.score_quantiles
        ):
            raise ValueError("score quantiles must be unique and strictly inside 0-1.")
        if self.weights != FROZEN_WEIGHTS:
            raise ValueError("weights must match the frozen documented kernel.")

    @property
    def weights(self) -> dict[str, float]:
        """Return a fresh mapping of the frozen weights."""
        return {
            "current_rack": self.current_rack_weight,
            "returned_1_15": self.returned_1_15_weight,
            "returned_16_50": self.returned_16_50_weight,
            "returned_51_100": self.returned_51_100_weight,
            "returned_over_100": self.returned_over_100_weight,
        }


@dataclass(frozen=True, slots=True)
class ScoreBandCutpoints:
    """Frozen numerical cutpoints derived from score-only development data."""

    q10: float
    q30: float
    q70: float
    q90: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoundMonetaryObservation:
    """Minimal validation-round record needed by streaming analyses."""

    score: float
    band: ScoreBand
    box_net: float
    initial_wager: float
    action_wager: float
    total_wager: float
    player_blackjacks: int
    double_actions: int
    split_actions: int
    outcome: Literal["win", "loss", "push"]
    cards_consumed: int
    initial_cards: tuple[CardIndicators, CardIndicators, CardIndicators]


@dataclass(slots=True)
class BandAccumulator:
    """Streaming monetary, event, and initial-composition totals for one band."""

    rounds: int = 0
    initial_wager: float = 0.0
    action_wager: float = 0.0
    total_wager: float = 0.0
    player_net: float = 0.0
    round_net_sq: float = 0.0
    player_blackjacks: int = 0
    double_actions: int = 0
    split_actions: int = 0
    winning_rounds: int = 0
    losing_rounds: int = 0
    push_rounds: int = 0
    cards_consumed: int = 0
    player_first_hi_lo: int = 0
    dealer_upcard_hi_lo: int = 0
    player_second_hi_lo: int = 0
    player_low: int = 0
    player_ten_value: int = 0
    player_ace: int = 0
    dealer_low: int = 0
    dealer_ten_value: int = 0
    dealer_ace: int = 0

    def add(self, observation: RoundMonetaryObservation) -> None:
        self.rounds += 1
        self.initial_wager += observation.initial_wager
        self.action_wager += observation.action_wager
        self.total_wager += observation.total_wager
        self.player_net += observation.box_net
        self.round_net_sq += observation.box_net**2
        self.player_blackjacks += observation.player_blackjacks
        self.double_actions += observation.double_actions
        self.split_actions += observation.split_actions
        self.winning_rounds += int(observation.outcome == "win")
        self.losing_rounds += int(observation.outcome == "loss")
        self.push_rounds += int(observation.outcome == "push")
        self.cards_consumed += observation.cards_consumed
        first, dealer, second = observation.initial_cards
        self.player_first_hi_lo += first.hi_lo
        self.dealer_upcard_hi_lo += dealer.hi_lo
        self.player_second_hi_lo += second.hi_lo
        self.player_low += first.low + second.low
        self.player_ten_value += first.ten_value + second.ten_value
        self.player_ace += first.ace + second.ace
        self.dealer_low += dealer.low
        self.dealer_ten_value += dealer.ten_value
        self.dealer_ace += dealer.ace

    def merge(self, other: BandAccumulator) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_metrics(self) -> dict[str, int | float | None]:
        rounds = self.rounds
        variance = (
            (self.round_net_sq - self.player_net**2 / rounds) / (rounds - 1)
            if rounds > 1
            else None
        )
        standard_deviation = sqrt(max(0.0, variance)) if variance is not None else None
        return {
            "rounds": rounds,
            "initial_wager": self.initial_wager,
            "additional_action_wager": self.action_wager,
            "total_wager": self.total_wager,
            "total_player_net": self.player_net,
            "total_casino_net": -self.player_net,
            "player_edge_per_initial_wager": (
                self.player_net / self.initial_wager if self.initial_wager else None
            ),
            "player_edge_per_total_wager": (
                self.player_net / self.total_wager if self.total_wager else None
            ),
            "mean_net_per_round": self.player_net / rounds if rounds else None,
            "round_net_sample_standard_deviation": standard_deviation,
            "naive_round_standard_error": (
                standard_deviation / sqrt(rounds)
                if standard_deviation is not None
                else None
            ),
            "player_blackjack_count": self.player_blackjacks,
            "player_blackjack_rate": self.player_blackjacks / rounds
            if rounds
            else None,
            "double_actions": self.double_actions,
            "double_action_rate": self.double_actions / rounds if rounds else None,
            "split_actions": self.split_actions,
            "split_action_rate": self.split_actions / rounds if rounds else None,
            "winning_rounds": self.winning_rounds,
            "winning_round_rate": self.winning_rounds / rounds if rounds else None,
            "losing_rounds": self.losing_rounds,
            "losing_round_rate": self.losing_rounds / rounds if rounds else None,
            "push_rounds": self.push_rounds,
            "push_round_rate": self.push_rounds / rounds if rounds else None,
            "average_cards_consumed": self.cards_consumed / rounds if rounds else None,
            "player_first_hi_lo_mean": (
                self.player_first_hi_lo / rounds if rounds else None
            ),
            "dealer_upcard_hi_lo_mean": (
                self.dealer_upcard_hi_lo / rounds if rounds else None
            ),
            "player_second_hi_lo_mean": (
                self.player_second_hi_lo / rounds if rounds else None
            ),
            "combined_initial_hi_lo_mean": (
                (
                    self.player_first_hi_lo
                    + self.dealer_upcard_hi_lo
                    + self.player_second_hi_lo
                )
                / (3 * rounds)
                if rounds
                else None
            ),
            "player_card_low_rate": self.player_low / (2 * rounds) if rounds else None,
            "player_card_ten_value_rate": (
                self.player_ten_value / (2 * rounds) if rounds else None
            ),
            "player_card_ace_rate": self.player_ace / (2 * rounds) if rounds else None,
            "dealer_upcard_low_rate": self.dealer_low / rounds if rounds else None,
            "dealer_upcard_ten_value_rate": (
                self.dealer_ten_value / rounds if rounds else None
            ),
            "dealer_upcard_ace_rate": self.dealer_ace / rounds if rounds else None,
        }


@dataclass(slots=True)
class DecisionFrequencyRecorder:
    """Source-blind delegating strategy that records ordinary decision states."""

    strategy: CasinoBlackjackStrategy
    counts: dict[tuple[str, ...], int]
    source: Source = "physical_iid"
    band: ScoreBand = "neutral"

    def choose_action(self, *, decision: BlackjackDecisionState) -> GameAction:
        action = self.strategy.choose_action(decision=decision)
        category_type, category_value = decision_hand_category(decision.player_ranks)
        legal_actions = "|".join(
            sorted(action_type.value for action_type in decision.legal_actions)
        )
        key = (
            self.source,
            self.band,
            decision.dealer_upcard_rank,
            category_type,
            category_value,
            str(decision.is_split_hand).lower(),
            legal_actions,
        )
        self.counts[key] = self.counts.get(key, 0) + 1
        return action


def linear_quantile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated sample quantile."""
    if not values:
        raise ValueError("quantiles require at least one score.")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly inside zero and one.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def freeze_score_cutpoints(
    scores: Sequence[float], quantiles: Sequence[float]
) -> ScoreBandCutpoints:
    """Construct cutpoints from score scalars only; no outcomes are accepted."""
    if len(quantiles) != 4:
        raise ValueError("exactly four quantiles are required.")
    values = tuple(linear_quantile(scores, quantile) for quantile in quantiles)
    if tuple(sorted(values)) != values:
        raise RuntimeError("development cutpoints are not ordered.")
    return ScoreBandCutpoints(*values)


def assign_score_band(score: float, cutpoints: ScoreBandCutpoints) -> ScoreBand:
    """Assign one and only one frozen score band."""
    if score <= cutpoints.q10:
        return "strong_high_rich"
    if score <= cutpoints.q30:
        return "moderate_high_rich"
    if score <= cutpoints.q70:
        return "neutral"
    if score <= cutpoints.q90:
        return "moderate_low_rich"
    return "strong_low_rich"


def decision_hand_category(player_ranks: Sequence[Rank]) -> tuple[str, str]:
    """Return broad pair/soft/hard decision-state category fields."""
    if len(player_ranks) == 2:
        left = split_value_from_rank(player_ranks[0])
        right = split_value_from_rank(player_ranks[1])
        if left == right:
            return "pair", str(left)
    total = 0
    ace_count = 0
    for rank in player_ranks:
        if rank == "A":
            total += 1
            ace_count += 1
        elif rank in {"T", "J", "Q", "K"}:
            total += 10
        else:
            total += int(rank)
    is_soft = ace_count > 0 and total + 10 <= 21
    if is_soft:
        total += 10
    return ("soft_total" if is_soft else "hard_total", str(total))


def run_conditional_profitability_experiment(
    config: ConditionalProfitabilityConfig,
) -> dict[str, Any]:
    """Run score-only development followed by held-out monetary validation."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    development_scores: list[float] = []
    for seed in config.development_seeds:
        development_scores.extend(_collect_development_scores(config, seed))
    cutpoints = freeze_score_cutpoints(development_scores, config.score_quantiles)
    frozen_cutpoint_record = cutpoints.as_dict()

    continuous_per_seed: list[dict[str, Any]] = []
    placebo_rows: list[dict[str, Any]] = []
    band_per_seed: list[dict[str, Any]] = []
    decision_counts: dict[tuple[str, ...], int] = {}

    for source in SOURCE_NAMES:
        for seed in config.validation_seeds:
            result = _run_validation_seed(
                config,
                source,
                seed,
                cutpoints,
                decision_counts,
            )
            continuous_per_seed.append(result["continuous"])
            placebo_rows.append(result["placebo"])
            band_per_seed.extend(result["bands"])
            if cutpoints.as_dict() != frozen_cutpoint_record:
                raise RuntimeError("validation mutated frozen score cutpoints.")

    continuous = _aggregate_continuous_slopes(continuous_per_seed)
    paired_slopes = _paired_monetary_slopes(
        continuous_per_seed, config.validation_seeds
    )
    placebo = _aggregate_placebo_rows(placebo_rows)
    band_aggregate = _aggregate_band_metrics(band_per_seed)
    band_differences = _band_source_differences(band_per_seed, config.validation_seeds)
    contrasts = _score_band_contrasts(band_per_seed, config.validation_seeds)
    ordered_trends = _ordered_band_trends(band_per_seed, config.validation_seeds)
    candidate_states = _evaluate_all_candidate_states(
        band_aggregate,
        band_differences,
        total_one2six_rounds=(
            len(config.validation_seeds) * config.validation_rounds_per_seed
        ),
    )
    decision_rows = _decision_frequency_rows(decision_counts)
    decision_summary = _decision_frequency_summary(decision_rows)
    plot_paths = _write_plots(
        config.output_dir,
        band_aggregate,
        continuous_per_seed,
        continuous,
        placebo_rows,
    )

    config_payload = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "frozen_weights": config.weights,
        "band_score_direction": (
            "lower is predicted high-rich; higher is predicted low-rich"
        ),
        "cutpoint_construction_inputs": ["predicted_hi_lo_shift"],
        "monetary_outcomes_used_for_cutpoints": False,
        "permutation_seed_offset": PERMUTATION_SEED_OFFSET,
    }
    cutpoint_payload = {
        "source": "one2six",
        "development_seeds": list(config.development_seeds),
        "development_score_count": len(development_scores),
        "quantiles": list(config.score_quantiles),
        "cutpoints": frozen_cutpoint_record,
        "monetary_outcomes_used": False,
    }
    summary = {
        "experiment": "heldout_conditional_profitability",
        "config": config_payload,
        "score_band_cutpoints": cutpoint_payload,
        "continuous_monetary_slopes": continuous,
        "continuous_monetary_slopes_per_seed": continuous_per_seed,
        "paired_source_monetary_slopes": paired_slopes,
        "permutation_placebo_slopes": placebo,
        "permutation_placebo_slopes_per_seed": placebo_rows,
        "score_band_profitability": band_aggregate,
        "per_seed_score_band_profitability": band_per_seed,
        "score_band_source_differences": band_differences,
        "score_band_contrasts": contrasts,
        "ordered_band_trends": ordered_trends,
        "candidate_positive_ev_states": candidate_states,
        "decision_state_frequency_summary": decision_summary,
        "hidden_state_exported": False,
        "plot_paths": plot_paths,
    }
    _validate_privacy(summary)
    _write_outputs(
        config.output_dir,
        summary,
        config_payload,
        cutpoint_payload,
        decision_rows,
    )
    return summary


def _new_trajectory(
    config: ConditionalProfitabilityConfig,
    source_name: SourceName,
    seed: int,
) -> tuple[LedgerCardSource, CasinoBlackjackGame, PublishedApproxCasinoStrategy]:
    ledger = ObservableCohortLedger()
    source = LedgerCardSource(
        source=_make_source(source_name, config.deck_count, seed),
        ledger=ledger,
    )
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            base_bet=config.base_bet,
            box_count=1,
            box_bets={1: config.base_bet},
            deck_count=config.deck_count,
        )
    )
    return source, game, PublishedApproxCasinoStrategy()


def _burn_in(
    *,
    rounds: int,
    source: LedgerCardSource,
    game: CasinoBlackjackGame,
    strategy: CasinoBlackjackStrategy,
) -> int:
    round_index = 0
    for _ in range(rounds):
        game.play_round(
            round_index=round_index,
            card_source=source,
            strategy=strategy,
        )
        round_index += 1
    return round_index


def _frozen_state(
    config: ConditionalProfitabilityConfig,
    source: LedgerCardSource,
    game: CasinoBlackjackGame,
) -> FadingState:
    current_rack = CohortCounts.from_cards(game.pending_discard_rack)
    returned_by_band = source.ledger.active_by_band(source.draw_count)
    return calculate_fading_state(
        current_rack=current_rack,
        returned_by_band=returned_by_band,
        weights=config.weights,
    )


def _collect_development_scores(
    config: ConditionalProfitabilityConfig, seed: int
) -> list[float]:
    """Collect only One2Six frozen scores; round outcomes are never inspected."""
    source, game, strategy = _new_trajectory(config, "one2six", seed)
    round_index = _burn_in(
        rounds=config.burn_in_rounds,
        source=source,
        game=game,
        strategy=strategy,
    )
    scores: list[float] = []
    for _ in range(config.development_rounds_per_seed):
        scores.append(_frozen_state(config, source, game).predicted_hi_lo_shift)
        game.play_round(
            round_index=round_index,
            card_source=source,
            strategy=strategy,
        )
        round_index += 1
    if isinstance(source.source, One2SixCardSource):
        source.source.assert_invariants(game.pending_discard_rack)
    return scores


def _run_validation_seed(
    config: ConditionalProfitabilityConfig,
    source_name: Source,
    seed: int,
    cutpoints: ScoreBandCutpoints,
    decision_counts: dict[tuple[str, ...], int],
) -> dict[str, Any]:
    source, game, fixed_strategy = _new_trajectory(config, source_name, seed)
    recorder = DecisionFrequencyRecorder(fixed_strategy, decision_counts)
    recorder.source = source_name
    round_index = _burn_in(
        rounds=config.burn_in_rounds,
        source=source,
        game=game,
        strategy=fixed_strategy,
    )
    regression = RegressionDiagnostics()
    score_stats = RunningStats()
    scores: list[float] = []
    monetary_outcomes: list[float] = []
    bands: dict[ScoreBand, BandAccumulator] = {
        band: BandAccumulator() for band in SCORE_BANDS
    }

    for _ in range(config.validation_rounds_per_seed):
        state = _frozen_state(config, source, game)
        score = state.predicted_hi_lo_shift
        band = assign_score_band(score, cutpoints)
        recorder.band = band
        draw_count_before = source.draw_count
        table, initial_cards = _play_round_capture_initial(
            game,
            source,
            recorder,
            round_index,
        )
        observation = _round_observation(
            score=score,
            band=band,
            table=table,
            initial_cards=initial_cards,
            cards_consumed=source.draw_count - draw_count_before,
        )
        normalized_net = observation.box_net / observation.initial_wager
        regression.add(score, normalized_net)
        score_stats.add(score)
        scores.append(score)
        monetary_outcomes.append(normalized_net)
        bands[band].add(observation)
        round_index += 1

    if isinstance(source.source, One2SixCardSource):
        source.source.assert_invariants(game.pending_discard_rack)
    continuous = {
        "source": source_name,
        "seed": seed,
        **regression.as_dict(),
        "predictor_mean": score_stats.as_dict()["mean"],
        "minimum_score": score_stats.minimum,
        "maximum_score": score_stats.maximum,
    }
    placebo_regression = deterministic_permutation_regression(
        scores,
        monetary_outcomes,
        seed=seed,
    )
    placebo = {
        "source": source_name,
        "seed": seed,
        "actual_slope": continuous["slope"],
        "placebo_slope": placebo_regression["slope"],
        "sample_count": placebo_regression["sample_count"],
        "placebo_correlation": placebo_regression["correlation"],
    }
    band_rows = []
    for band in SCORE_BANDS:
        if bands[band].rounds == 0:
            continue
        band_rows.append(
            {
                "source": source_name,
                "seed": seed,
                "score_band": band,
                "band_order": BAND_SCORES[band],
                "round_frequency": (
                    bands[band].rounds / config.validation_rounds_per_seed
                ),
                **bands[band].as_metrics(),
            }
        )
    if (
        sum(int(row["rounds"]) for row in band_rows)
        != config.validation_rounds_per_seed
    ):
        raise RuntimeError("validation rounds do not reconcile across score bands.")
    return {
        "continuous": continuous,
        "placebo": placebo,
        "bands": band_rows,
    }


def _round_observation(
    *,
    score: float,
    band: ScoreBand,
    table: TableState,
    initial_cards: Sequence[Any],
    cards_consumed: int,
) -> RoundMonetaryObservation:
    box = table.boxes[0]
    initial_wager = box.base_bet
    total_wager = sum(hand.wager * (2 if hand.is_doubled else 1) for hand in box.hands)
    box_net = sum(hand.net_result for hand in box.hands)
    indicators = tuple(card_indicators(card.rank) for card in initial_cards)
    if len(indicators) != 3:
        raise RuntimeError("one-box initial deal must contain exactly three cards.")
    return RoundMonetaryObservation(
        score=score,
        band=band,
        box_net=box_net,
        initial_wager=initial_wager,
        action_wager=total_wager - initial_wager,
        total_wager=total_wager,
        player_blackjacks=sum(
            is_natural_blackjack(
                hand.cards,
                blackjack_eligible=hand.blackjack_eligible,
            )
            for hand in box.hands
        ),
        double_actions=sum(hand.is_doubled for hand in box.hands),
        split_actions=int(len(box.hands) > 1),
        outcome="win" if box_net > 0 else "loss" if box_net < 0 else "push",
        cards_consumed=cards_consumed,
        initial_cards=(indicators[0], indicators[1], indicators[2]),
    )


def deterministic_permutation_regression(
    scores: Sequence[float], outcomes: Sequence[float], *, seed: int
) -> dict[str, Any]:
    """Regress outcomes on one deterministic within-seed score permutation."""
    if len(scores) != len(outcomes):
        raise ValueError("score and outcome lengths must match.")
    permuted = list(scores)
    Random(seed + PERMUTATION_SEED_OFFSET).shuffle(permuted)
    regression = RegressionDiagnostics()
    for score, outcome in zip(permuted, outcomes, strict=True):
        regression.add(score, outcome)
    return {**regression.as_dict(), "permuted_scores": tuple(permuted)}


def _aggregate_continuous_slopes(
    per_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for source in SOURCE_NAMES:
        matching = [row for row in per_seed if row["source"] == source]
        slopes = [float(row["slope"]) for row in matching if row["slope"] is not None]
        summary = student_t_summary(slopes)
        rows.append(
            {
                "source": source,
                "contributing_seeds": len(slopes),
                "mean_monetary_slope": summary["mean"],
                "sample_standard_deviation": summary["sample_standard_deviation"],
                "standard_error": summary["standard_error"],
                "student_t_95_ci": summary["student_t_95_ci"],
                "minimum": summary["minimum"],
                "maximum": summary["maximum"],
                "negative_seed_slopes": sum(value < 0 for value in slopes),
                "positive_seed_slopes": sum(value > 0 for value in slopes),
                "expected_advantage_sign": "negative",
            }
        )
    return rows


def _paired_monetary_slopes(
    per_seed: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    differences = []
    by_seed = {}
    for seed in seeds:
        values = {
            str(row["source"]): row["slope"] for row in per_seed if row["seed"] == seed
        }
        if values.get("physical_iid") is None or values.get("one2six") is None:
            continue
        difference = float(values["one2six"]) - float(values["physical_iid"])
        differences.append(difference)
        by_seed[str(seed)] = difference
    summary = student_t_summary(differences)
    return [
        {
            "metric": "one2six_minus_physical_iid_monetary_slope",
            "contributing_seeds": len(differences),
            "mean_difference": summary["mean"],
            "sample_standard_deviation": summary["sample_standard_deviation"],
            "standard_error": summary["standard_error"],
            "student_t_95_ci": summary["student_t_95_ci"],
            "minimum": summary["minimum"],
            "maximum": summary["maximum"],
            "negative_seeds": sum(value < 0 for value in differences),
            "positive_seeds": sum(value > 0 for value in differences),
            "differences_by_seed": by_seed,
            "expected_advantage_sign": "negative",
        }
    ]


def _aggregate_placebo_rows(
    per_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for source in SOURCE_NAMES:
        matching = [row for row in per_seed if row["source"] == source]
        for slope_type in ("actual_slope", "placebo_slope"):
            slopes = [
                float(row[slope_type])
                for row in matching
                if row[slope_type] is not None
            ]
            summary = student_t_summary(slopes)
            rows.append(
                {
                    "source": source,
                    "slope_type": slope_type,
                    "contributing_seeds": len(slopes),
                    "mean_slope": summary["mean"],
                    "student_t_95_ci": summary["student_t_95_ci"],
                    "negative_seed_slopes": sum(value < 0 for value in slopes),
                    "positive_seed_slopes": sum(value > 0 for value in slopes),
                }
            )
    return rows


def _aggregate_band_metrics(
    per_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    derived_fields = (
        "round_frequency",
        "rounds",
        "initial_wager",
        "additional_action_wager",
        "total_wager",
        "total_player_net",
        "total_casino_net",
        *MONETARY_FIELDS,
        "player_blackjack_count",
        "double_actions",
        "split_actions",
        "winning_rounds",
        "losing_rounds",
        "push_rounds",
        *EVENT_FIELDS,
        *COMPOSITION_FIELDS,
    )
    for source in SOURCE_NAMES:
        for band in SCORE_BANDS:
            matching = [
                row
                for row in per_seed
                if row["source"] == source and row["score_band"] == band
            ]
            if not matching:
                continue
            pooled = BandAccumulator()
            for item in matching:
                pooled.merge(_accumulator_from_row(item))
            pooled_metrics = pooled.as_metrics()
            aggregate_row: dict[str, Any] = {
                "source": source,
                "score_band": band,
                "band_order": BAND_SCORES[band],
                "contributing_seeds": len(matching),
                "round_frequency": _mean(
                    [float(item["round_frequency"]) for item in matching]
                ),
                **pooled_metrics,
            }
            for field_name in derived_fields:
                values = [
                    float(item[field_name])
                    for item in matching
                    if item.get(field_name) is not None
                ]
                summary = student_t_summary(values)
                aggregate_row[f"mean_seed_{field_name}"] = summary["mean"]
                aggregate_row[f"{field_name}_seed_ci"] = summary["student_t_95_ci"]
                aggregate_row[f"positive_seed_{field_name}"] = sum(
                    value > 0 for value in values
                )
                aggregate_row[f"negative_seed_{field_name}"] = sum(
                    value < 0 for value in values
                )
            rows.append(aggregate_row)
    return rows


def _accumulator_from_row(row: Mapping[str, Any]) -> BandAccumulator:
    """Reconstruct mergeable totals from one per-seed metrics row."""
    rounds = int(row["rounds"])
    mean_net = float(row["mean_net_per_round"])
    standard_deviation = row["round_net_sample_standard_deviation"]
    round_net_sq = rounds * mean_net**2
    if standard_deviation is not None and rounds > 1:
        round_net_sq += (rounds - 1) * float(standard_deviation) ** 2
    return BandAccumulator(
        rounds=rounds,
        initial_wager=float(row["initial_wager"]),
        action_wager=float(row["additional_action_wager"]),
        total_wager=float(row["total_wager"]),
        player_net=float(row["total_player_net"]),
        round_net_sq=round_net_sq,
        player_blackjacks=int(row["player_blackjack_count"]),
        double_actions=int(row["double_actions"]),
        split_actions=int(row["split_actions"]),
        winning_rounds=int(row["winning_rounds"]),
        losing_rounds=int(row["losing_rounds"]),
        push_rounds=int(row["push_rounds"]),
        cards_consumed=round(float(row["average_cards_consumed"]) * rounds),
        player_first_hi_lo=round(float(row["player_first_hi_lo_mean"]) * rounds),
        dealer_upcard_hi_lo=round(float(row["dealer_upcard_hi_lo_mean"]) * rounds),
        player_second_hi_lo=round(float(row["player_second_hi_lo_mean"]) * rounds),
        player_low=round(float(row["player_card_low_rate"]) * 2 * rounds),
        player_ten_value=round(float(row["player_card_ten_value_rate"]) * 2 * rounds),
        player_ace=round(float(row["player_card_ace_rate"]) * 2 * rounds),
        dealer_low=round(float(row["dealer_upcard_low_rate"]) * rounds),
        dealer_ten_value=round(float(row["dealer_upcard_ten_value_rate"]) * rounds),
        dealer_ace=round(float(row["dealer_upcard_ace_rate"]) * rounds),
    )


def _band_source_differences(
    per_seed: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for band in SCORE_BANDS:
        differences = []
        for seed in seeds:
            values = {
                str(row["source"]): row["player_edge_per_initial_wager"]
                for row in per_seed
                if row["seed"] == seed and row["score_band"] == band
            }
            if values.get("physical_iid") is None or values.get("one2six") is None:
                continue
            differences.append(float(values["one2six"]) - float(values["physical_iid"]))
        summary = student_t_summary(differences)
        rows.append(
            {
                "score_band": band,
                "contributing_seed_pairs": len(differences),
                "mean_paired_edge_difference": summary["mean"],
                "student_t_95_ci": summary["student_t_95_ci"],
                "positive_seeds": sum(value > 0 for value in differences),
                "negative_seeds": sum(value < 0 for value in differences),
            }
        )
    return rows


def _score_band_contrasts(
    per_seed: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    per_source_seed: dict[tuple[Source, int, str], float] = {}
    for source in SOURCE_NAMES:
        for seed in seeds:
            edge_by_band = {
                row["score_band"]: float(row["player_edge_per_initial_wager"])
                for row in per_seed
                if row["source"] == source and row["seed"] == seed
            }
            for contrast, (left, right) in CONTRASTS.items():
                if left in edge_by_band and right in edge_by_band:
                    per_source_seed[(source, seed, contrast)] = (
                        edge_by_band[left] - edge_by_band[right]
                    )
    rows = []
    for contrast in CONTRASTS:
        for source in SOURCE_NAMES:
            values = [
                per_source_seed[(source, seed, contrast)]
                for seed in seeds
                if (source, seed, contrast) in per_source_seed
            ]
            rows.append(_summary_row(source, contrast, values))
        differences = [
            per_source_seed[("one2six", seed, contrast)]
            - per_source_seed[("physical_iid", seed, contrast)]
            for seed in seeds
            if ("one2six", seed, contrast) in per_source_seed
            and ("physical_iid", seed, contrast) in per_source_seed
        ]
        rows.append(_summary_row("one2six_minus_physical_iid", contrast, differences))
    return rows


def _ordered_band_trends(
    per_seed: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    seed_rows = []
    for source in SOURCE_NAMES:
        for seed in seeds:
            regression = RegressionDiagnostics()
            matching = [
                row
                for row in per_seed
                if row["source"] == source and row["seed"] == seed
            ]
            for row in matching:
                regression.add(
                    float(row["band_order"]),
                    float(row["player_edge_per_initial_wager"]),
                )
            diagnostics = regression.as_dict()
            seed_rows.append(
                {
                    "row_scope": "per_seed",
                    "source": source,
                    "seed": seed,
                    "player_edge_ordered_band_slope": diagnostics["slope"],
                    "advantage_direction_slope": (
                        -float(diagnostics["slope"])
                        if diagnostics["slope"] is not None
                        else None
                    ),
                    "sample_count": diagnostics["sample_count"],
                }
            )
    rows = list(seed_rows)
    for source in SOURCE_NAMES:
        slopes = [
            float(row["player_edge_ordered_band_slope"])
            for row in seed_rows
            if row["source"] == source
            and row["player_edge_ordered_band_slope"] is not None
        ]
        rows.append(
            {
                "row_scope": "aggregate",
                **_summary_row(source, "ordered_band_player_edge", slopes),
                "expected_player_edge_sign": "negative",
                "expected_advantage_direction_sign": "positive",
            }
        )
    differences = []
    for seed in seeds:
        values = {
            str(row["source"]): row["player_edge_ordered_band_slope"]
            for row in seed_rows
            if row["seed"] == seed
        }
        if values.get("physical_iid") is not None and values.get("one2six") is not None:
            differences.append(float(values["one2six"]) - float(values["physical_iid"]))
    rows.append(
        {
            "row_scope": "aggregate",
            **_summary_row(
                "one2six_minus_physical_iid",
                "ordered_band_player_edge",
                differences,
            ),
            "expected_player_edge_sign": "negative",
            "expected_advantage_direction_sign": "positive",
        }
    )
    return rows


def _summary_row(source: str, metric: str, values: Sequence[float]) -> dict[str, Any]:
    summary = student_t_summary(list(values))
    return {
        "source": source,
        "metric": metric,
        "contributing_seeds": len(values),
        "mean": summary["mean"],
        "sample_standard_deviation": summary["sample_standard_deviation"],
        "standard_error": summary["standard_error"],
        "student_t_95_ci": summary["student_t_95_ci"],
        "minimum": summary["minimum"],
        "maximum": summary["maximum"],
        "negative_seeds": sum(value < 0 for value in values),
        "positive_seeds": sum(value > 0 for value in values),
    }


def evaluate_candidate_state(
    *,
    one2six_band: Mapping[str, Any],
    iid_band: Mapping[str, Any],
    paired_difference: Mapping[str, Any],
    total_one2six_rounds: int,
) -> dict[str, Any]:
    """Apply the six pre-specified candidate positive-EV gate conditions."""
    mean_edge = one2six_band.get("mean_seed_player_edge_per_initial_wager")
    interval = one2six_band.get("player_edge_per_initial_wager_seed_ci")
    positive_seeds = int(
        one2six_band.get("positive_seed_player_edge_per_initial_wager", 0)
    )
    frequency = int(one2six_band.get("rounds", 0)) / total_one2six_rounds
    iid_mean = iid_band.get("mean_seed_player_edge_per_initial_wager")
    iid_interval = iid_band.get("player_edge_per_initial_wager_seed_ci")
    difference_interval = paired_difference.get("student_t_95_ci")
    conditions = {
        "positive_mean_edge": mean_edge is not None and float(mean_edge) > 0,
        "positive_seed_ci": _ci_positive(interval),
        "at_least_nine_positive_seeds": positive_seeds >= 9,
        "at_least_five_percent_frequency": frequency >= 0.05,
        "physical_iid_not_same_positive_effect": not (
            iid_mean is not None and float(iid_mean) > 0 and _ci_positive(iid_interval)
        ),
        "positive_paired_source_difference": _ci_positive(difference_interval),
    }
    return {
        "score_band": one2six_band["score_band"],
        "candidate_positive_ev_state": all(conditions.values()),
        "conditions": conditions,
        "one2six_round_frequency": frequency,
    }


def _evaluate_all_candidate_states(
    aggregate_bands: Sequence[Mapping[str, Any]],
    differences: Sequence[Mapping[str, Any]],
    *,
    total_one2six_rounds: int,
) -> list[dict[str, Any]]:
    rows = []
    for band in SCORE_BANDS:
        one = next(
            row
            for row in aggregate_bands
            if row["source"] == "one2six" and row["score_band"] == band
        )
        iid = next(
            row
            for row in aggregate_bands
            if row["source"] == "physical_iid" and row["score_band"] == band
        )
        difference = next(row for row in differences if row["score_band"] == band)
        rows.append(
            evaluate_candidate_state(
                one2six_band=one,
                iid_band=iid,
                paired_difference=difference,
                total_one2six_rounds=total_one2six_rounds,
            )
        )
    return rows


def _decision_frequency_rows(
    counts: Mapping[tuple[str, ...], int],
) -> list[dict[str, Any]]:
    rows = []
    for key, count in sorted(counts.items()):
        (
            source,
            band,
            dealer_upcard,
            category_type,
            category_value,
            split_indicator,
            legal_actions,
        ) = key
        rows.append(
            {
                "source": source,
                "score_band": band,
                "dealer_upcard": dealer_upcard,
                "hand_category_type": category_type,
                "hand_category_value": category_value,
                "is_split_hand": split_indicator,
                "legal_actions": legal_actions,
                "decision_count": count,
            }
        )
    return rows


def _decision_frequency_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for source in SOURCE_NAMES:
        for band in SCORE_BANDS:
            matching = [
                row
                for row in rows
                if row["source"] == source and row["score_band"] == band
            ]
            summary.append(
                {
                    "source": source,
                    "score_band": band,
                    "decision_state_cells": len(matching),
                    "decision_count": sum(
                        int(row["decision_count"]) for row in matching
                    ),
                }
            )
    return summary


def _write_plots(
    output_dir: Path,
    aggregate_bands: Sequence[Mapping[str, Any]],
    continuous_per_seed: Sequence[Mapping[str, Any]],
    continuous_aggregate: Sequence[Mapping[str, Any]],
    placebo_per_seed: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    combined_name = "player_edge_by_score_band.png"
    plot_conditional_edge_by_band(
        aggregate_bands,
        source=None,
        output_path=output_dir / combined_name,
    )
    paths["player_edge_by_score_band"] = combined_name
    for source in ("one2six", "physical_iid"):
        name = f"player_edge_by_score_band_{source}.png"
        plot_conditional_edge_by_band(
            aggregate_bands,
            source=source,
            output_path=output_dir / name,
        )
        paths[f"player_edge_by_score_band_{source}"] = name
    slope_name = "monetary_slope_by_seed.png"
    plot_monetary_slopes_by_seed(
        continuous_per_seed,
        continuous_aggregate,
        output_path=output_dir / slope_name,
    )
    paths["monetary_slope_by_seed"] = slope_name
    placebo_name = "actual_vs_placebo_monetary_slope.png"
    plot_actual_vs_placebo_slopes(
        placebo_per_seed,
        output_path=output_dir / placebo_name,
    )
    paths["actual_vs_placebo_monetary_slope"] = placebo_name
    composition_name = "initial_deal_composition_by_score_band.png"
    plot_initial_composition_by_band(
        aggregate_bands,
        output_path=output_dir / composition_name,
    )
    paths["initial_deal_composition_by_score_band"] = composition_name
    frequency_name = "score_band_frequency.png"
    plot_score_band_frequency(
        aggregate_bands,
        output_path=output_dir / frequency_name,
    )
    paths["score_band_frequency"] = frequency_name
    return paths


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    cutpoint_payload: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "experiment_config.json", config_payload)
    _write_json(output_dir / "score_band_cutpoints.json", cutpoint_payload)
    _write_csv(
        output_dir / "continuous_monetary_slopes.csv",
        [
            *summary["continuous_monetary_slopes_per_seed"],
            *summary["continuous_monetary_slopes"],
        ],
    )
    _write_csv(
        output_dir / "paired_source_monetary_slopes.csv",
        summary["paired_source_monetary_slopes"],
    )
    _write_csv(
        output_dir / "permutation_placebo_slopes.csv",
        [
            *summary["permutation_placebo_slopes_per_seed"],
            *summary["permutation_placebo_slopes"],
        ],
    )
    profitability_fields = {
        "source",
        "score_band",
        "band_order",
        "contributing_seeds",
        "rounds",
        "round_frequency",
        "initial_wager",
        "additional_action_wager",
        "total_wager",
        "total_player_net",
        "total_casino_net",
        *MONETARY_FIELDS,
    }
    event_fields = {
        "source",
        "score_band",
        "band_order",
        "contributing_seeds",
        "rounds",
        "player_blackjack_count",
        "double_actions",
        "split_actions",
        "winning_rounds",
        "losing_rounds",
        "push_rounds",
        *EVENT_FIELDS,
    }
    composition_fields = {
        "source",
        "score_band",
        "band_order",
        "contributing_seeds",
        "rounds",
        *COMPOSITION_FIELDS,
    }
    frequency_fields = {
        "source",
        "score_band",
        "band_order",
        "contributing_seeds",
        "rounds",
        "round_frequency",
    }
    aggregate_bands = summary["score_band_profitability"]
    _write_csv(
        output_dir / "score_band_profitability.csv",
        [
            *_select_metric_fields(aggregate_bands, profitability_fields),
            *summary["score_band_source_differences"],
        ],
    )
    _write_csv(
        output_dir / "per_seed_score_band_profitability.csv",
        summary["per_seed_score_band_profitability"],
    )
    _write_csv(
        output_dir / "score_band_contrasts.csv",
        summary["score_band_contrasts"],
    )
    _write_csv(
        output_dir / "ordered_band_trends.csv",
        summary["ordered_band_trends"],
    )
    _write_csv(
        output_dir / "score_band_initial_deal_composition.csv",
        _select_metric_fields(aggregate_bands, composition_fields),
    )
    _write_csv(
        output_dir / "score_band_event_rates.csv",
        _select_metric_fields(aggregate_bands, event_fields),
    )
    _write_csv(
        output_dir / "score_band_frequency.csv",
        _select_metric_fields(aggregate_bands, frequency_fields),
    )
    _write_csv(
        output_dir / "decision_state_frequency.csv",
        decision_rows,
    )
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _select_metric_fields(
    rows: Sequence[Mapping[str, Any]], base_fields: set[str]
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        fields = {
            key: value
            for key, value in row.items()
            if key in base_fields
            or any(
                key.startswith(f"mean_seed_{field}")
                or key.startswith(f"positive_seed_{field}")
                or key.startswith(f"negative_seed_{field}")
                or key == f"{field}_seed_ci"
                for field in base_fields
            )
        }
        selected.append(fields)
    return selected


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    cutpoints = summary["score_band_cutpoints"]["cutpoints"]
    lines = [
        "This experiment tests whether the frozen observable fading-exclusion",
        "score predicts next-round blackjack profitability.",
        "",
        "Unlike the preceding composition-response studies, monetary EV is the",
        "primary endpoint.",
        "",
        "The fixed player strategy is unchanged.",
        "",
        "No betting threshold, action deviation, box-count policy or live",
        "advantage strategy is selected in this experiment.",
        "",
        "# Held-Out Conditional Profitability",
        "",
        "## Frozen Score and Bands",
        "",
        "Weights: current rack 1.00; returned 1-15 0.75; returned 16-50 0.40; ",
        "returned 51-100 0.20; older 0.00.",
        "",
        f"Cutpoints: q10={_fmt(cutpoints['q10'])}, q30={_fmt(cutpoints['q30'])}, "
        f"q70={_fmt(cutpoints['q70'])}, q90={_fmt(cutpoints['q90'])}.",
        "",
        "## Continuous Monetary Response",
        "",
        "| Source | Mean monetary slope | 95% seed CI | Expected sign | Sign count |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["continuous_monetary_slopes"]:
        lines.append(
            f"| {_source_label(row['source'])} | {_fmt(row['mean_monetary_slope'])} | "
            f"{_interval(row['student_t_95_ci'])} | negative | "
            f"-{row['negative_seed_slopes']}/+{row['positive_seed_slopes']} |"
        )
    paired = summary["paired_source_monetary_slopes"][0]
    lines.extend(
        [
            "",
            "## Direct Source Difference",
            "",
            "| Metric | One2Six minus IID | 95% CI | Sign count |",
            "|---|---:|---:|---:|",
            f"| Monetary slope | {_fmt(paired['mean_difference'])} | "
            f"{_interval(paired['student_t_95_ci'])} | "
            f"-{paired['negative_seeds']}/+{paired['positive_seeds']} |",
            "",
            "## Conditional Edge by Score Band",
            "",
            "| Score band | Physical IID edge | One2Six edge | One2Six minus IID |",
            "|---|---:|---:|---:|",
        ]
    )
    for band in SCORE_BANDS:
        iid = _band_row(summary, "physical_iid", band)
        one = _band_row(summary, "one2six", band)
        difference = next(
            row
            for row in summary["score_band_source_differences"]
            if row["score_band"] == band
        )
        lines.append(
            f"| {band} | {_fmt(iid['mean_seed_player_edge_per_initial_wager'])} | "
            f"{_fmt(one['mean_seed_player_edge_per_initial_wager'])} | "
            f"{_fmt(difference['mean_paired_edge_difference'])} |"
        )
    lines.extend(["", "## Opportunity Frequency", ""])
    for band in SCORE_BANDS:
        iid = _band_row(summary, "physical_iid", band)
        one = _band_row(summary, "one2six", band)
        lines.append(
            f"- {band}: Physical IID {_pct(iid['round_frequency'])}; "
            f"One2Six {_pct(one['round_frequency'])}."
        )
    lines.extend(["", "## Pre-Specified Contrasts", ""])
    for row in summary["score_band_contrasts"]:
        lines.append(
            f"- {_source_label(row['source'])} {row['metric']}: "
            f"{_fmt(row['mean'])}, CI {_interval(row['student_t_95_ci'])}."
        )
    lines.extend(["", "## Ordered Monotonic Trend", ""])
    for row in summary["ordered_band_trends"]:
        if row["row_scope"] != "aggregate":
            continue
        lines.append(
            f"- {_source_label(row['source'])}: player-edge slope "
            f"{_fmt(row['mean'])}, CI {_interval(row['student_t_95_ci'])}, "
            f"-{row['negative_seeds']}/+{row['positive_seeds']} seeds."
        )
    lines.extend(["", "## Deterministic Permutation Placebo", ""])
    for row in summary["permutation_placebo_slopes"]:
        lines.append(
            f"- {_source_label(row['source'])} {row['slope_type']}: "
            f"{_fmt(row['mean_slope'])}, CI {_interval(row['student_t_95_ci'])}."
        )
    lines.extend(["", "## Initial-Deal Mechanism Check", ""])
    for band in SCORE_BANDS:
        one = _band_row(summary, "one2six", band)
        lines.append(
            f"- {band}: initial Hi-Lo "
            f"{_fmt(one['mean_seed_combined_initial_hi_lo_mean'])}, player low "
            f"{_pct(one['mean_seed_player_card_low_rate'])}, player ten-value "
            f"{_pct(one['mean_seed_player_card_ten_value_rate'])}, player ace "
            f"{_pct(one['mean_seed_player_card_ace_rate'])}."
        )
    lines.extend(["", "## Event Rates", ""])
    for band in SCORE_BANDS:
        one = _band_row(summary, "one2six", band)
        lines.append(
            f"- {band}: blackjack {_pct(one['mean_seed_player_blackjack_rate'])}, "
            f"double {_pct(one['mean_seed_double_action_rate'])}, split "
            f"{_pct(one['mean_seed_split_action_rate'])}, win "
            f"{_pct(one['mean_seed_winning_round_rate'])}."
        )
    candidates = [
        row
        for row in summary["candidate_positive_ev_states"]
        if row["candidate_positive_ev_state"]
    ]
    lines.extend(
        [
            "",
            "## Candidate Advantage-State Verdict",
            "",
            (
                "Candidate positive-EV states: "
                + ", ".join(row["score_band"] for row in candidates)
                if candidates
                else "No validated positive-EV state under the fixed strategy."
            ),
            "",
            "## Strategy-Readiness Verdict",
            "",
            _strategy_readiness_verdict(summary, bool(candidates)),
            "",
            "The ordered-band player-edge slope is reported with its mathematically "
            "correct expected negative sign; its sign-flipped advantage-direction "
            "diagnostic has the requested expected positive sign.",
            "",
            "## Plots",
            "",
        ]
    )
    for label, path in summary["plot_paths"].items():
        lines.append(f"- [{label}]({path})")
    lines.append("")
    return "\n".join(lines)


def _strategy_readiness_verdict(summary: Mapping[str, Any], has_candidate: bool) -> str:
    one = next(
        row
        for row in summary["continuous_monetary_slopes"]
        if row["source"] == "one2six"
    )
    slope_predicts = _ci_negative(one["student_t_95_ci"])
    one_bands = [
        row for row in summary["score_band_profitability"] if row["source"] == "one2six"
    ]
    closest = max(
        one_bands,
        key=lambda row: float(row["mean_seed_player_edge_per_initial_wager"]),
    )
    close_to_break_even = (
        abs(float(closest["mean_seed_player_edge_per_initial_wager"])) <= 0.01
    )
    return (
        f"Continuous profitability prediction={slope_predicts}; validated positive-EV "
        f"band={has_candidate}; best fixed-strategy band={closest['score_band']} at "
        f"{_pct(closest['mean_seed_player_edge_per_initial_wager'])}; close enough to "
        f"break-even for later action-deviation analysis={close_to_break_even}. "
        "No action optimization was performed."
    )


def _band_row(
    summary: Mapping[str, Any], source: str, band: ScoreBand
) -> Mapping[str, Any]:
    return next(
        row
        for row in summary["score_band_profitability"]
        if row["source"] == source and row["score_band"] == band
    )


def _validate_privacy(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True).lower()
    matches = [term for term in PRIVATE_TERMS if term in serialized]
    if matches:
        raise RuntimeError(f"hidden source fields reached exports: {matches}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _ci_positive(interval: object) -> bool:
    return isinstance(interval, list) and len(interval) == 2 and interval[0] > 0


def _ci_negative(interval: object) -> bool:
    return isinstance(interval, list) and len(interval) == 2 and interval[1] < 0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _pct(value: object) -> str:
    return "NA" if value is None else f"{100 * float(value):.3f}%"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "NA"
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _source_label(source: object) -> str:
    labels = {
        "physical_iid": "Physical IID",
        "one2six": "One2Six",
        "one2six_minus_physical_iid": "One2Six minus IID",
    }
    return labels.get(str(source), str(source))
