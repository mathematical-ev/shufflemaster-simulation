# SPDX-License-Identifier: GPL-3.0-or-later

"""Held-out validation of a frozen observable fading-exclusion signal."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Final, Literal

from experiments.multi_box_counterfactual import (
    CardSourceType,
    SourceName,
    _make_source,
)
from experiments.observable_card_response import (
    BASELINE_PROBABILITIES,
    OUTCOMES,
    CardIndicators,
    OutcomeName,
    RegressionAccumulator,
    card_indicators,
    rack_features,
)
from experiments.plots import (
    plot_component_contributions,
    plot_heldout_calibration,
    plot_response_slopes,
    plot_score_distribution,
    plot_score_group_edges,
)
from experiments.single_box_game_validation import student_t_summary
from shufflemaster_sim.card_sources import CardSource, One2SixCardSource
from shufflemaster_sim.cards import Card
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.hand_values import is_natural_blackjack
from shufflemaster_sim.state import TableState
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)

DEVELOPMENT_SEEDS: Final[frozenset[int]] = frozenset({42, 43, 44, 45, 46})
SOURCE_NAMES: Final[tuple[SourceName, ...]] = ("physical_iid", "one2six")
COMPONENT_NAMES: Final[tuple[str, ...]] = (
    "current_rack",
    "returned_1_15",
    "returned_16_50",
    "returned_51_100",
)
FROZEN_WEIGHTS: Final[dict[str, float]] = {
    "current_rack": 1.00,
    "returned_1_15": 0.75,
    "returned_16_50": 0.40,
    "returned_51_100": 0.20,
    "returned_over_100": 0.00,
}
SCORE_GROUPS: Final[tuple[str, ...]] = (
    "predicted_high_rich",
    "near_neutral",
    "predicted_low_rich",
)
PRIVATE_TERMS: Final[tuple[str, ...]] = (
    "physical_id",
    "draw_id",
    "shelf_id",
    "buffer_contents",
    "feeder_contents",
    "carousel_contents",
    "rng_state",
    "source_snapshot",
)


@dataclass(frozen=True, slots=True)
class FadingExclusionValidationConfig:
    """Configuration for held-out frozen-kernel validation."""

    seeds: tuple[int, ...] = (47, 48, 49, 50, 51)
    deck_count: int = 6
    base_bet: float = 10.0
    rounds_per_seed: int = 50_000
    burn_in_rounds: int = 1_000
    probe_states_per_seed: int = 3_000
    probe_cards: int = 15
    current_rack_weight: float = 1.00
    returned_1_15_weight: float = 0.75
    returned_16_50_weight: float = 0.40
    returned_51_100_weight: float = 0.20
    returned_over_100_weight: float = 0.00
    allow_weight_override: bool = False
    output_dir: Path = Path("experiments/outputs/fading_exclusion_validation_heldout")

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one held-out seed must be supplied.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique.")
        if DEVELOPMENT_SEEDS.intersection(self.seeds):
            raise ValueError("held-out seeds must be disjoint from development seeds.")
        if self.deck_count != 6:
            raise ValueError("fading validation requires exactly six decks.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        if self.rounds_per_seed <= 0 or self.probe_states_per_seed <= 0:
            raise ValueError("round and probe counts must be positive.")
        if self.probe_states_per_seed > self.rounds_per_seed:
            raise ValueError("probe_states_per_seed cannot exceed rounds_per_seed.")
        if self.burn_in_rounds < 0:
            raise ValueError("burn_in_rounds must be non-negative.")
        if self.probe_cards != 15:
            raise ValueError("probe_cards must equal 15.")
        weights = self.weights
        if not self.allow_weight_override and weights != FROZEN_WEIGHTS:
            raise ValueError("weights must match the frozen documented kernel.")
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("cohort weights must be non-negative.")

    @property
    def weights(self) -> dict[str, float]:
        """Return the configured immutable age-band weights."""
        return {
            "current_rack": self.current_rack_weight,
            "returned_1_15": self.returned_1_15_weight,
            "returned_16_50": self.returned_16_50_weight,
            "returned_51_100": self.returned_51_100_weight,
            "returned_over_100": self.returned_over_100_weight,
        }


@dataclass(frozen=True, slots=True)
class CohortCounts:
    """Observable card counts for one rack or returned cohort."""

    card_count: int
    hi_lo: int
    low: int
    neutral: int
    ten_value: int
    ace: int

    @classmethod
    def from_cards(cls, cards: Sequence[Card]) -> CohortCounts:
        features = rack_features(cards)
        return cls(
            card_count=features.rack_size,
            hi_lo=features.rack_hi_lo_count,
            low=features.rack_low_count,
            neutral=features.rack_neutral_count,
            ten_value=features.rack_ten_value_count,
            ace=features.rack_ace_count,
        )


@dataclass(frozen=True, slots=True)
class ReturnedCohort:
    """Observable returned batch with absolute first-future draw index."""

    return_draw_index: int
    counts: CohortCounts


@dataclass(slots=True)
class ObservableCohortLedger:
    """Persistent observable batch ledger for one source/seed trajectory."""

    returned: list[ReturnedCohort] = field(default_factory=list)

    def record_return(self, *, return_draw_index: int, cards: Sequence[Card]) -> None:
        """Record only observable composition and timing."""
        self.returned.append(
            ReturnedCohort(
                return_draw_index=return_draw_index,
                counts=CohortCounts.from_cards(cards),
            )
        )

    def active_by_band(
        self, current_draw_index: int
    ) -> dict[str, list[ReturnedCohort]]:
        """Assign each active cohort to exactly one dealt-card age band."""
        bands = {name: [] for name in COMPONENT_NAMES[1:]}
        retained: list[ReturnedCohort] = []
        for cohort in self.returned:
            age = current_draw_index - cohort.return_draw_index
            band = cohort_age_band(age)
            if band is not None:
                bands[band].append(cohort)
                retained.append(cohort)
        self.returned = retained
        return bands


@dataclass(slots=True)
class LedgerCardSource:
    """Card-source wrapper that updates the observable cohort ledger."""

    source: CardSourceType
    ledger: ObservableCohortLedger

    @property
    def draw_count(self) -> int:
        return self.source.draw_count

    def before_round(self) -> None:
        self.source.before_round()

    def draw_card(self) -> Card:
        return self.source.draw_card()

    def accept_discards(self, cards: Sequence[Card]) -> None:
        self.ledger.record_return(
            return_draw_index=self.source.draw_count,
            cards=cards,
        )
        self.source.accept_discards(cards)


@dataclass(frozen=True, slots=True)
class FadingState:
    """Frozen observable score calculated before the next initial deal."""

    effective_weighted_card_count: float
    weighted_hi_lo_count: float
    weighted_low_count: float
    weighted_neutral_count: float
    weighted_ten_value_count: float
    weighted_ace_count: float
    weighted_low_excess: float
    weighted_neutral_excess: float
    weighted_ten_value_excess: float
    weighted_ace_excess: float
    effective_remaining_cards: float
    predicted_hi_lo_shift: float
    predicted_low_shift: float
    predicted_neutral_shift: float
    predicted_ten_value_shift: float
    predicted_ace_shift: float
    contributions: Mapping[str, Mapping[str, float]]

    def predictor(self, outcome: OutcomeName) -> float:
        return float(getattr(self, f"predicted_{outcome}_shift"))


@dataclass(frozen=True, slots=True)
class ProbeObservation:
    state: FadingState
    cards: tuple[CardIndicators, ...]


@dataclass(frozen=True, slots=True)
class RoundObservation:
    state: FadingState
    initial_cards: tuple[CardIndicators, CardIndicators, CardIndicators]
    box_net: float
    initial_wager: float
    additional_wager: float
    total_wager: float
    blackjack: int
    doubles: int
    splits: int
    outcome: Literal["win", "loss", "push"]


@dataclass(slots=True)
class RegressionDiagnostics:
    """Regression sufficient statistics with correlation/calibration diagnostics."""

    ols: RegressionAccumulator = field(default_factory=RegressionAccumulator)
    sum_yy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.ols.add(x, y)
        self.sum_yy += y * y

    def merge(self, other: RegressionDiagnostics) -> None:
        self.ols.merge(other.ols)
        self.sum_yy += other.sum_yy

    def as_dict(self) -> dict[str, Any]:
        n = self.ols.count
        mean_x = self.ols.sum_x / n if n else None
        mean_y = self.ols.sum_y / n if n else None
        var_x = (self.ols.sum_xx - self.ols.sum_x**2 / n) / (n - 1) if n > 1 else None
        var_y = (self.sum_yy - self.ols.sum_y**2 / n) / (n - 1) if n > 1 else None
        covariance = (
            (self.ols.sum_xy - self.ols.sum_x * self.ols.sum_y / n) / (n - 1)
            if n > 1
            else None
        )
        correlation = (
            covariance / sqrt(var_x * var_y)
            if covariance is not None and var_x and var_y and var_x > 0 and var_y > 0
            else None
        )
        calibration_rmse = (
            sqrt(
                max(
                    0.0,
                    (self.sum_yy + self.ols.sum_xx - 2 * self.ols.sum_xy) / n,
                )
            )
            if n
            else None
        )
        return {
            "sample_count": n,
            "slope": self.ols.slope(),
            "intercept": self.ols.intercept(),
            "predictor_standard_deviation": sqrt(var_x) if var_x is not None else None,
            "outcome_mean": mean_y,
            "correlation": correlation,
            "calibration_bias": mean_y - mean_x if mean_x is not None else None,
            "calibration_rmse": calibration_rmse,
        }


@dataclass(slots=True)
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def merge(self, other: RunningStats) -> None:
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq
        if other.minimum is not None:
            self.minimum = (
                other.minimum
                if self.minimum is None
                else min(self.minimum, other.minimum)
            )
            self.maximum = (
                other.maximum
                if self.maximum is None
                else max(self.maximum, other.maximum)
            )

    def as_dict(self) -> dict[str, Any]:
        mean = self.total / self.count if self.count else None
        variance = (
            (self.total_sq - self.total**2 / self.count) / (self.count - 1)
            if self.count > 1
            else None
        )
        return {
            "count": self.count,
            "mean": mean,
            "sample_standard_deviation": sqrt(max(0.0, variance))
            if variance is not None
            else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(slots=True)
class ComponentDiagnostics:
    """Four-component sufficient statistics for one outcome."""

    count: int = 0
    sums: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in COMPONENT_NAMES}
    )
    sums_sq: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in COMPONENT_NAMES}
    )
    nonzero: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in COMPONENT_NAMES}
    )
    cross: dict[tuple[str, str], float] = field(default_factory=dict)

    def add(self, values: Mapping[str, float]) -> None:
        self.count += 1
        for name in COMPONENT_NAMES:
            value = values[name]
            self.sums[name] += value
            self.sums_sq[name] += value * value
            self.nonzero[name] += int(value != 0)
        for index, left in enumerate(COMPONENT_NAMES):
            for right in COMPONENT_NAMES[index + 1 :]:
                self.cross[(left, right)] = (
                    self.cross.get((left, right), 0.0) + values[left] * values[right]
                )

    def merge(self, other: ComponentDiagnostics) -> None:
        self.count += other.count
        for name in COMPONENT_NAMES:
            self.sums[name] += other.sums[name]
            self.sums_sq[name] += other.sums_sq[name]
            self.nonzero[name] += other.nonzero[name]
        for pair, value in other.cross.items():
            self.cross[pair] = self.cross.get(pair, 0.0) + value

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"states": self.count}
        for name in COMPONENT_NAMES:
            mean = self.sums[name] / self.count if self.count else None
            variance = (
                (self.sums_sq[name] - self.sums[name] ** 2 / self.count)
                / (self.count - 1)
                if self.count > 1
                else None
            )
            row[f"mean_{name}"] = mean
            row[f"sd_{name}"] = (
                sqrt(max(0.0, variance)) if variance is not None else None
            )
            row[f"nonzero_proportion_{name}"] = (
                self.nonzero[name] / self.count if self.count else None
            )
        for (left, right), cross_sum in self.cross.items():
            row[f"correlation_{left}__{right}"] = _correlation_from_sums(
                self.count,
                self.sums[left],
                self.sums[right],
                self.sums_sq[left],
                self.sums_sq[right],
                cross_sum,
            )
        return row


def cohort_age_band(age: int) -> str | None:
    """Return the unique frozen band; age zero uses the freshest returned band."""
    if age < 0:
        raise ValueError("cohort age cannot be negative.")
    if age <= 15:
        return "returned_1_15"
    if age <= 50:
        return "returned_16_50"
    if age <= 100:
        return "returned_51_100"
    return None


def calculate_fading_state(
    *,
    current_rack: CohortCounts,
    returned_by_band: Mapping[str, Sequence[ReturnedCohort]],
    weights: Mapping[str, float],
) -> FadingState:
    """Calculate the frozen weighted exclusion index from observable cohorts."""
    raw: dict[str, CohortCounts] = {"current_rack": current_rack}
    for band in COMPONENT_NAMES[1:]:
        cohorts = returned_by_band.get(band, ())
        raw[band] = _sum_cohorts(cohorts)
    contributions: dict[str, dict[str, float]] = {}
    for name in COMPONENT_NAMES:
        weight = weights[name]
        counts = raw[name]
        contributions[name] = {
            "card_count": weight * counts.card_count,
            "hi_lo": weight * counts.hi_lo,
            "low": weight * counts.low,
            "neutral": weight * counts.neutral,
            "ten_value": weight * counts.ten_value,
            "ace": weight * counts.ace,
        }
    totals = {
        field_name: sum(component[field_name] for component in contributions.values())
        for field_name in ("card_count", "hi_lo", "low", "neutral", "ten_value", "ace")
    }
    remaining = 312.0 - totals["card_count"]
    if remaining <= 0:
        raise RuntimeError("effective_remaining_cards must be positive.")
    excess = {
        outcome: totals[outcome]
        - BASELINE_PROBABILITIES[outcome] * totals["card_count"]
        for outcome in ("low", "neutral", "ten_value", "ace")
    }
    return FadingState(
        effective_weighted_card_count=totals["card_count"],
        weighted_hi_lo_count=totals["hi_lo"],
        weighted_low_count=totals["low"],
        weighted_neutral_count=totals["neutral"],
        weighted_ten_value_count=totals["ten_value"],
        weighted_ace_count=totals["ace"],
        weighted_low_excess=excess["low"],
        weighted_neutral_excess=excess["neutral"],
        weighted_ten_value_excess=excess["ten_value"],
        weighted_ace_excess=excess["ace"],
        effective_remaining_cards=remaining,
        predicted_hi_lo_shift=-totals["hi_lo"] / remaining,
        predicted_low_shift=-excess["low"] / remaining,
        predicted_neutral_shift=-excess["neutral"] / remaining,
        predicted_ten_value_shift=-excess["ten_value"] / remaining,
        predicted_ace_shift=-excess["ace"] / remaining,
        contributions=contributions,
    )


def run_fading_exclusion_validation(
    config: FadingExclusionValidationConfig,
) -> dict[str, Any]:
    """Run held-out source comparisons using the frozen observable kernel."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_primary: list[dict[str, Any]] = []
    per_seed_position: list[dict[str, Any]] = []
    per_seed_initial: list[dict[str, Any]] = []
    per_seed_monetary: list[dict[str, Any]] = []
    score_group_rows: list[dict[str, Any]] = []
    probe_count_rows: list[dict[str, Any]] = []
    distribution_acc: dict[tuple[SourceName, OutcomeName], RunningStats] = defaultdict(
        RunningStats
    )
    component_acc: dict[tuple[SourceName, OutcomeName], ComponentDiagnostics] = (
        defaultdict(ComponentDiagnostics)
    )
    score_values: dict[SourceName, list[float]] = defaultdict(list)

    for source_name in SOURCE_NAMES:
        for seed in config.seeds:
            result = _run_source_seed(config, source_name, seed)
            per_seed_primary.extend(result["primary"])
            per_seed_position.extend(result["positions"])
            per_seed_initial.extend(result["initial"])
            per_seed_monetary.extend(result["monetary"])
            score_group_rows.extend(result["score_groups"])
            probe_count_rows.append(
                {
                    "source": source_name,
                    "seed": seed,
                    "probe_states": result["probe_count"],
                }
            )
            score_values[source_name].extend(result["score_values"])
            for outcome in OUTCOMES:
                distribution_acc[(source_name, outcome)].merge(
                    result["distribution"][outcome]
                )
                component_acc[(source_name, outcome)].merge(
                    result["components"][outcome]
                )

    primary = _aggregate_regression_rows(
        per_seed_primary, key_fields=("source", "outcome")
    )
    positions = _aggregate_regression_rows(
        per_seed_position, key_fields=("source", "position", "outcome")
    )
    _mark_position_stability(positions)
    initial = _aggregate_regression_rows(
        per_seed_initial, key_fields=("source", "endpoint", "outcome", "predictor")
    )
    monetary = _aggregate_regression_rows(
        per_seed_monetary, key_fields=("source", "predictor")
    )
    paired = _paired_source_differences(per_seed_primary, config.seeds)
    score_groups = _aggregate_score_groups(score_group_rows)
    distribution_rows = [
        {"source": source, "score": outcome, **accumulator.as_dict()}
        for (source, outcome), accumulator in sorted(distribution_acc.items())
    ]
    component_rows = [
        {"source": source, "outcome": outcome, **accumulator.as_row()}
        for (source, outcome), accumulator in sorted(component_acc.items())
    ]
    plot_paths = _write_plots(
        config.output_dir,
        primary,
        positions,
        distribution_rows,
        component_rows,
        score_groups,
        score_values,
    )
    config_payload = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "development_seeds": sorted(DEVELOPMENT_SEEDS),
        "held_out_seeds": list(config.seeds),
        "frozen_weights": config.weights,
        "probe_schedule": "floor(i * rounds_per_seed / probe_states_per_seed)",
    }
    summary = {
        "experiment": "heldout_fading_exclusion_validation",
        "config": config_payload,
        "hidden_state_exported": False,
        "next15_primary_validation": primary,
        "next15_per_seed_validation": per_seed_primary,
        "next15_position_validation": positions,
        "paired_source_slope_differences": paired,
        "actual_initial_deal_validation": initial,
        "actual_round_monetary_response": monetary,
        "actual_round_score_groups": score_groups,
        "fading_state_distribution": distribution_rows,
        "fading_state_components": component_rows,
        "probe_state_counts": probe_count_rows,
        "plot_paths": plot_paths,
    }
    _validate_privacy(summary)
    _write_json(config.output_dir / "summary.json", summary)
    _write_json(config.output_dir / "experiment_config.json", config_payload)
    _write_csv(config.output_dir / "fading_state_distribution.csv", distribution_rows)
    _write_csv(config.output_dir / "fading_state_components.csv", component_rows)
    _write_csv(config.output_dir / "next15_primary_validation.csv", primary)
    _write_csv(config.output_dir / "next15_per_seed_validation.csv", per_seed_primary)
    _write_csv(config.output_dir / "next15_position_validation.csv", positions)
    _write_csv(config.output_dir / "paired_source_slope_differences.csv", paired)
    _write_csv(config.output_dir / "actual_initial_deal_validation.csv", initial)
    _write_csv(config.output_dir / "actual_round_monetary_response.csv", monetary)
    _write_csv(config.output_dir / "actual_round_score_groups.csv", score_groups)
    _write_csv(config.output_dir / "probe_state_counts.csv", probe_count_rows)
    (config.output_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    return summary


def _run_source_seed(
    config: FadingExclusionValidationConfig,
    source_name: SourceName,
    seed: int,
) -> dict[str, Any]:
    ledger = ObservableCohortLedger()
    wrapped = LedgerCardSource(
        _make_source(source_name, config.deck_count, seed), ledger
    )
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            base_bet=config.base_bet,
            box_count=1,
            box_bets={1: config.base_bet},
            deck_count=config.deck_count,
        )
    )
    strategy = PublishedApproxCasinoStrategy()
    round_index = 0
    for _ in range(config.burn_in_rounds):
        game.play_round(round_index=round_index, card_source=wrapped, strategy=strategy)
        round_index += 1

    probe_indices = {
        index * config.rounds_per_seed // config.probe_states_per_seed
        for index in range(config.probe_states_per_seed)
    }
    probes: list[ProbeObservation] = []
    rounds: list[RoundObservation] = []
    distribution = {outcome: RunningStats() for outcome in OUTCOMES}
    components = {outcome: ComponentDiagnostics() for outcome in OUTCOMES}
    score_values: list[float] = []

    for validation_index in range(config.rounds_per_seed):
        current_rack = CohortCounts.from_cards(game.pending_discard_rack)
        bands = ledger.active_by_band(wrapped.draw_count)
        state = calculate_fading_state(
            current_rack=current_rack,
            returned_by_band=bands,
            weights=config.weights,
        )
        for outcome in OUTCOMES:
            distribution[outcome].add(state.predictor(outcome))
            components[outcome].add(
                {name: state.contributions[name][outcome] for name in COMPONENT_NAMES}
            )
        score_values.append(state.predicted_hi_lo_shift)
        if validation_index in probe_indices:
            original_draw_count = wrapped.draw_count
            original_rack = game.pending_discard_rack
            probe_source = deepcopy(wrapped.source)
            probe_cards = tuple(
                probe_source.draw_card() for _ in range(config.probe_cards)
            )
            if (
                wrapped.draw_count != original_draw_count
                or game.pending_discard_rack != original_rack
            ):
                raise RuntimeError("Held-out probe mutated the continuous trajectory.")
            if source_name == "one2six":
                probe_source.assert_invariants([*original_rack, *probe_cards])
            probes.append(
                ProbeObservation(
                    state=state,
                    cards=tuple(card_indicators(card.rank) for card in probe_cards),
                )
            )
        table, initial_cards = _play_round_capture_initial(
            game, wrapped, strategy, round_index
        )
        rounds.append(_round_observation(state, table, initial_cards))
        round_index += 1

    if len(probes) != config.probe_states_per_seed:
        raise RuntimeError("Deterministic probe schedule did not produce exact count.")
    if isinstance(wrapped.source, One2SixCardSource):
        wrapped.source.assert_invariants(game.pending_discard_rack)
    return {
        "primary": _probe_seed_rows(source_name, seed, probes, position=None),
        "positions": [
            row
            for position in range(1, 16)
            for row in _probe_seed_rows(source_name, seed, probes, position=position)
        ],
        "initial": _initial_seed_rows(source_name, seed, rounds),
        "monetary": _monetary_seed_rows(source_name, seed, rounds),
        "score_groups": _score_group_seed_rows(source_name, seed, rounds),
        "probe_count": len(probes),
        "distribution": distribution,
        "components": components,
        "score_values": score_values,
    }


def _play_round_capture_initial(
    game: CasinoBlackjackGame,
    source: CardSource,
    strategy: PublishedApproxCasinoStrategy,
    round_index: int,
) -> tuple[TableState, tuple[Card, Card, Card]]:
    source.before_round()
    game.burn_initial_card(source)
    table = game.create_table(round_index)
    game.deal_initial_cards(table, source)
    hand = table.boxes[0].hands[0]
    initial = (hand.cards[0], table.dealer.cards[0], hand.cards[1])
    game.settle_immediate_blackjacks(table)
    game.play_player_hands(table, source, strategy)
    game.play_dealer(table, source)
    game.settle(table)
    game.collect_remaining_layout_cards(table)
    game.stage_discard_rack_for_next_round(table)
    return table, initial


def _round_observation(
    state: FadingState,
    table: TableState,
    initial_cards: tuple[Card, Card, Card],
) -> RoundObservation:
    box = table.boxes[0]
    initial_wager = box.base_bet
    total_wager = sum(hand.wager * (2 if hand.is_doubled else 1) for hand in box.hands)
    net = sum(hand.net_result for hand in box.hands)
    return RoundObservation(
        state=state,
        initial_cards=tuple(card_indicators(card.rank) for card in initial_cards),  # type: ignore[arg-type]
        box_net=net,
        initial_wager=initial_wager,
        additional_wager=total_wager - initial_wager,
        total_wager=total_wager,
        blackjack=sum(
            is_natural_blackjack(hand.cards, blackjack_eligible=hand.blackjack_eligible)
            for hand in box.hands
        ),
        doubles=sum(hand.is_doubled for hand in box.hands),
        splits=int(len(box.hands) > 1),
        outcome="win" if net > 0 else "loss" if net < 0 else "push",
    )


def _probe_seed_rows(
    source: SourceName,
    seed: int,
    probes: Sequence[ProbeObservation],
    *,
    position: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for outcome in OUTCOMES:
        regression = RegressionDiagnostics()
        for probe in probes:
            values = [float(getattr(card, outcome)) for card in probe.cards]
            observed = (
                values[position - 1]
                if position is not None
                else sum(values) / len(values)
            )
            regression.add(
                probe.state.predictor(outcome),
                observed - BASELINE_PROBABILITIES.get(outcome, 0.0),
            )
        rows.append(
            {
                "source": source,
                "seed": seed,
                "position": position,
                "outcome": outcome,
                **regression.as_dict(),
            }
        )
    return rows


def _initial_seed_rows(
    source: SourceName, seed: int, rounds: Sequence[RoundObservation]
) -> list[dict[str, Any]]:
    endpoints: dict[str, tuple[tuple[int, ...], OutcomeName]] = {
        "initial_mean_hi_lo": ((0, 1, 2), "hi_lo"),
        "initial_low_rate": ((0, 1, 2), "low"),
        "initial_ten_value_rate": ((0, 1, 2), "ten_value"),
        "initial_ace_rate": ((0, 1, 2), "ace"),
        "player_card_ace_rate": ((0, 2), "ace"),
        "dealer_upcard_ace_rate": ((1,), "ace"),
        "player_card_ten_value_rate": ((0, 2), "ten_value"),
        "dealer_upcard_ten_value_rate": ((1,), "ten_value"),
        "dealer_upcard_low_rate": ((1,), "low"),
        "dealer_upcard_neutral_rate": ((1,), "neutral"),
    }
    rows = []
    for endpoint, (positions, outcome) in endpoints.items():
        regression = RegressionDiagnostics()
        for observation in rounds:
            observed = sum(
                float(getattr(observation.initial_cards[index], outcome))
                for index in positions
            ) / len(positions)
            regression.add(
                observation.state.predictor(outcome),
                observed - BASELINE_PROBABILITIES.get(outcome, 0.0),
            )
        rows.append(
            {
                "source": source,
                "seed": seed,
                "endpoint": endpoint,
                "outcome": outcome,
                "predictor": f"predicted_{outcome}_shift",
                **regression.as_dict(),
            }
        )
    for predictor_outcome in ("ten_value", "ace"):
        regression = RegressionDiagnostics()
        for observation in rounds:
            first, _, second = observation.initial_cards
            player_blackjack = float(
                (first.ace and second.ten_value) or (first.ten_value and second.ace)
            )
            regression.add(
                observation.state.predictor(predictor_outcome),
                player_blackjack,
            )
        rows.append(
            {
                "source": source,
                "seed": seed,
                "endpoint": "player_blackjack_rate",
                "outcome": "player_blackjack",
                "predictor": f"predicted_{predictor_outcome}_shift",
                **regression.as_dict(),
            }
        )
    return rows


def _monetary_seed_rows(
    source: SourceName, seed: int, rounds: Sequence[RoundObservation]
) -> list[dict[str, Any]]:
    rows = []
    for outcome in ("hi_lo", "low", "ten_value", "ace"):
        regression = RegressionDiagnostics()
        for observation in rounds:
            regression.add(
                observation.state.predictor(outcome),
                observation.box_net / observation.initial_wager,
            )
        rows.append(
            {
                "source": source,
                "seed": seed,
                "predictor": f"predicted_{outcome}_shift",
                **regression.as_dict(),
            }
        )
    return rows


def _score_group_seed_rows(
    source: SourceName, seed: int, rounds: Sequence[RoundObservation]
) -> list[dict[str, Any]]:
    rows = []
    for group in SCORE_GROUPS:
        selected = [
            round_
            for round_ in rounds
            if score_group(round_.state.predicted_hi_lo_shift) == group
        ]
        initial = sum(round_.initial_wager for round_ in selected)
        net = sum(round_.box_net for round_ in selected)
        rows.append(
            {
                "source": source,
                "seed_or_aggregate": seed,
                "score_group": group,
                "rounds": len(selected),
                "initial_wager": initial,
                "additional_wager": sum(round_.additional_wager for round_ in selected),
                "total_wager": sum(round_.total_wager for round_ in selected),
                "net_player_result": net,
                "edge_per_initial_wager": net / initial if initial else None,
                "wins": sum(round_.outcome == "win" for round_ in selected),
                "losses": sum(round_.outcome == "loss" for round_ in selected),
                "pushes": sum(round_.outcome == "push" for round_ in selected),
                "blackjacks": sum(round_.blackjack for round_ in selected),
                "doubles": sum(round_.doubles for round_ in selected),
                "splits": sum(round_.splits for round_ in selected),
            }
        )
    return rows


def score_group(predicted_hi_lo_shift: float) -> str:
    if predicted_hi_lo_shift < -0.0025:
        return "predicted_high_rich"
    if predicted_hi_lo_shift > 0.0025:
        return "predicted_low_rich"
    return "near_neutral"


def _aggregate_regression_rows(
    per_seed: Sequence[Mapping[str, Any]], *, key_fields: Sequence[str]
) -> list[dict[str, Any]]:
    keys = sorted(
        {tuple(row[field] for field in key_fields) for row in per_seed}, key=str
    )
    rows = []
    for key in keys:
        matching = [
            row for row in per_seed if tuple(row[field] for field in key_fields) == key
        ]
        slopes = [float(row["slope"]) for row in matching if row["slope"] is not None]
        summary = student_t_summary(slopes)
        rows.append(
            {
                **dict(zip(key_fields, key, strict=True)),
                "contributing_seeds": summary["independent_seed_runs"],
                "mean_seed_slope": summary["mean"],
                "sample_standard_deviation": summary["sample_standard_deviation"],
                "standard_error": summary["standard_error"],
                "student_t_95_ci": summary["student_t_95_ci"],
                "minimum_seed_slope": summary["minimum"],
                "maximum_seed_slope": summary["maximum"],
                "positive_seed_slopes": sum(value > 0 for value in slopes),
                "negative_seed_slopes": sum(value < 0 for value in slopes),
                "mean_seed_correlation": _mean(
                    [
                        float(row["correlation"])
                        for row in matching
                        if row["correlation"] is not None
                    ]
                ),
                "mean_predictor_standard_deviation": _mean(
                    [
                        float(row["predictor_standard_deviation"])
                        for row in matching
                        if row["predictor_standard_deviation"] is not None
                    ]
                ),
            }
        )
    return rows


def _mark_position_stability(rows: list[dict[str, Any]]) -> None:
    """Mark secondary position findings that do not meet stability checks."""
    lookup = {(row["source"], row["position"], row["outcome"]): row for row in rows}
    for row in rows:
        reasons: list[str] = []
        if row["contributing_seeds"] < 5:
            reasons.append("fewer_than_five_seeds")
        predictor_sd = row["mean_predictor_standard_deviation"]
        if predictor_sd is None or predictor_sd <= 1e-12:
            reasons.append("inadequate_predictor_variance")
        interval = row["student_t_95_ci"]
        if interval is None:
            reasons.append("seed_interval_unavailable")
        elif _ci_contains_zero(interval):
            reasons.append("seed_interval_includes_zero")
        if row["source"] == "one2six":
            iid = lookup.get(("physical_iid", row["position"], row["outcome"]))
            one_mean = row["mean_seed_slope"]
            iid_mean = iid["mean_seed_slope"] if iid is not None else None
            if (
                iid is not None
                and iid_mean is not None
                and one_mean is not None
                and iid_mean * one_mean > 0
                and abs(iid_mean) >= 0.5 * abs(one_mean)
                and not _ci_contains_zero(iid["student_t_95_ci"])
            ):
                reasons.append("physical_iid_has_similar_effect")
        row["unstable"] = bool(reasons)
        row["instability_reasons"] = reasons


def _paired_source_differences(
    per_seed: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for outcome in OUTCOMES:
        differences = []
        by_seed = {}
        for seed in seeds:
            values = {
                row["source"]: row["slope"]
                for row in per_seed
                if row["seed"] == seed and row["outcome"] == outcome
            }
            if values.get("physical_iid") is None or values.get("one2six") is None:
                continue
            difference = float(values["one2six"]) - float(values["physical_iid"])
            differences.append(difference)
            by_seed[str(seed)] = difference
        summary = student_t_summary(differences)
        rows.append(
            {
                "outcome": outcome,
                "contributing_seeds": len(differences),
                "mean_paired_slope_difference": summary["mean"],
                "sample_standard_deviation": summary["sample_standard_deviation"],
                "standard_error": summary["standard_error"],
                "student_t_95_ci": summary["student_t_95_ci"],
                "minimum": summary["minimum"],
                "maximum": summary["maximum"],
                "positive_seeds": sum(value > 0 for value in differences),
                "negative_seeds": sum(value < 0 for value in differences),
                "differences_by_seed": by_seed,
            }
        )
    return rows


def _aggregate_score_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = list(rows)
    for source in SOURCE_NAMES:
        for group in SCORE_GROUPS:
            matching = [
                row
                for row in rows
                if row["source"] == source and row["score_group"] == group
            ]
            initial = sum(float(row["initial_wager"]) for row in matching)
            net = sum(float(row["net_player_result"]) for row in matching)
            seed_edges = [
                float(row["edge_per_initial_wager"])
                for row in matching
                if row["edge_per_initial_wager"] is not None
            ]
            edge_summary = student_t_summary(seed_edges)
            result.append(
                {
                    "source": source,
                    "seed_or_aggregate": "aggregate",
                    "score_group": group,
                    "rounds": sum(int(row["rounds"]) for row in matching),
                    "initial_wager": initial,
                    "additional_wager": sum(
                        float(row["additional_wager"]) for row in matching
                    ),
                    "total_wager": sum(float(row["total_wager"]) for row in matching),
                    "net_player_result": net,
                    "edge_per_initial_wager": net / initial if initial else None,
                    "mean_seed_edge_per_initial_wager": edge_summary["mean"],
                    "seed_edge_student_t_95_ci": edge_summary["student_t_95_ci"],
                    "wins": sum(int(row["wins"]) for row in matching),
                    "losses": sum(int(row["losses"]) for row in matching),
                    "pushes": sum(int(row["pushes"]) for row in matching),
                    "blackjacks": sum(int(row["blackjacks"]) for row in matching),
                    "doubles": sum(int(row["doubles"]) for row in matching),
                    "splits": sum(int(row["splits"]) for row in matching),
                }
            )
    return result


def _write_plots(
    output_dir: Path,
    primary: Sequence[Mapping[str, Any]],
    positions: Sequence[Mapping[str, Any]],
    distributions: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    score_groups: Sequence[Mapping[str, Any]],
    score_values: Mapping[SourceName, Sequence[float]],
) -> dict[str, str]:
    paths = {}
    for outcome in ("hi_lo", "low", "ten_value", "ace"):
        name = f"frozen_score_calibration_{outcome}.png"
        plot_heldout_calibration(
            [row for row in primary if row["outcome"] == outcome],
            title=f"Held-out frozen-score calibration: {outcome.replace('_', ' ')}",
            output_path=output_dir / name,
        )
        paths[f"frozen_score_calibration_{outcome}"] = name
    for outcome in ("hi_lo", "ten_value", "ace"):
        name = f"heldout_position_response_{outcome}.png"
        plot_response_slopes(
            [row for row in positions if row["outcome"] == outcome],
            x_field="position",
            x_values=list(range(1, 16)),
            title=f"Held-out position response: {outcome.replace('_', ' ')}",
            x_label="Future card position",
            output_path=output_dir / name,
            reference_one=True,
        )
        paths[f"heldout_position_response_{outcome}"] = name
    distribution_name = "fading_score_distribution.png"
    plot_score_distribution(score_values, output_path=output_dir / distribution_name)
    paths["fading_score_distribution"] = distribution_name
    component_name = "fading_component_contributions.png"
    plot_component_contributions(components, output_path=output_dir / component_name)
    paths["fading_component_contributions"] = component_name
    edge_name = "actual_round_edge_by_score_group.png"
    plot_score_group_edges(score_groups, output_path=output_dir / edge_name)
    paths["actual_round_edge_by_score_group"] = edge_name
    return paths


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "This is a held-out validation of a frozen observable fading-exclusion",
        "signal.",
        "",
        "Weights were fixed using earlier development results and were not",
        "retuned on these seeds.",
        "",
        "The primary endpoint is next-15 card composition.",
        "",
        "Full-round monetary response is secondary and exploratory.",
        "",
        "No betting or box-count policy is selected.",
        "",
        "# Held-Out Frozen Fading-Exclusion Validation",
        "",
        "## Frozen Kernel",
        "",
        "| Cohort | Weight |",
        "|---|---:|",
        "| Current rack | 1.00 |",
        "| Returned 1-15 | 0.75 |",
        "| Returned 16-50 | 0.40 |",
        "| Returned 51-100 | 0.20 |",
        "| Older | 0.00 |",
        "",
        "## Primary Held-Out Validation",
        "",
        "| Source | Outcome | Mean slope | 95% seed CI | Sign count |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["next15_primary_validation"]:
        lines.append(
            f"| {_source_label(row['source'])} | {row['outcome']} | "
            f"{_fmt(row['mean_seed_slope'])} | {_interval(row['student_t_95_ci'])} | "
            f"+{row['positive_seed_slopes']}/-{row['negative_seed_slopes']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Source Differences",
            "",
            "| Outcome | One2Six minus IID slope | 95% CI | Positive seeds |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["paired_source_slope_differences"]:
        lines.append(
            f"| {row['outcome']} | {_fmt(row['mean_paired_slope_difference'])} | "
            f"{_interval(row['student_t_95_ci'])} | {row['positive_seeds']} |"
        )
    lines.extend(["", "## Actual Initial Deal", ""])
    for row in summary["actual_initial_deal_validation"]:
        lines.append(
            f"- {_source_label(row['source'])} {row['endpoint']}: slope "
            f"{_fmt(row['mean_seed_slope'])}, CI {_interval(row['student_t_95_ci'])}."
        )
    lines.extend(["", "## Monetary Response", ""])
    lines.append(
        "Exploratory univariate response only; no threshold or policy was fitted."
    )
    for row in summary["actual_round_monetary_response"]:
        lines.append(
            f"- {_source_label(row['source'])} {row['predictor']}: slope "
            f"{_fmt(row['mean_seed_slope'])}, CI {_interval(row['student_t_95_ci'])}."
        )
    lines.extend(["", "## Verdict", "", _verdict(summary), "", "## Plots", ""])
    for label, path in summary["plot_paths"].items():
        lines.append(f"- [{label}]({path})")
    lines.append("")
    return "\n".join(lines)


def _verdict(summary: Mapping[str, Any]) -> str:
    one = [
        row
        for row in summary["next15_primary_validation"]
        if row["source"] == "one2six"
    ]
    iid = [
        row
        for row in summary["next15_primary_validation"]
        if row["source"] == "physical_iid"
    ]
    paired = summary["paired_source_slope_differences"]
    one_positive = all(_ci_positive(row["student_t_95_ci"]) for row in one)
    iid_null = all(_ci_contains_zero(row["student_t_95_ci"]) for row in iid)
    paired_positive = all(_ci_positive(row["student_t_95_ci"]) for row in paired)
    monetary = [
        row
        for row in summary["actual_round_monetary_response"]
        if row["source"] == "one2six"
    ]
    monetary_stable = any(
        row["student_t_95_ci"] is not None
        and not _ci_contains_zero(row["student_t_95_ci"])
        for row in monetary
    )
    advance_to_ev = one_positive and paired_positive
    return (
        f"Frozen score predicts held-out One2Six composition={one_positive}; "
        f"Physical IID remains null={iid_null}; paired differences are positive="
        f"{paired_positive}; sufficient to justify later exact rank-specific EV "
        f"weighting={advance_to_ev}; stable exploratory monetary response="
        f"{monetary_stable}. This validates signal direction only and does not "
        "validate a betting strategy."
    )


def _sum_cohorts(cohorts: Sequence[ReturnedCohort]) -> CohortCounts:
    return CohortCounts(
        card_count=sum(cohort.counts.card_count for cohort in cohorts),
        hi_lo=sum(cohort.counts.hi_lo for cohort in cohorts),
        low=sum(cohort.counts.low for cohort in cohorts),
        neutral=sum(cohort.counts.neutral for cohort in cohorts),
        ten_value=sum(cohort.counts.ten_value for cohort in cohorts),
        ace=sum(cohort.counts.ace for cohort in cohorts),
    )


def _correlation_from_sums(
    count: int,
    sum_x: float,
    sum_y: float,
    sum_xx: float,
    sum_yy: float,
    sum_xy: float,
) -> float | None:
    if count < 2:
        return None
    centered_x = sum_xx - sum_x**2 / count
    centered_y = sum_yy - sum_y**2 / count
    if centered_x <= 0 or centered_y <= 0:
        return None
    return (sum_xy - sum_x * sum_y / count) / sqrt(centered_x * centered_y)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ci_positive(interval: object) -> bool:
    return isinstance(interval, list) and len(interval) == 2 and interval[0] > 0


def _ci_contains_zero(interval: object) -> bool:
    return (
        isinstance(interval, list)
        and len(interval) == 2
        and interval[0] <= 0 <= interval[1]
    )


def _validate_privacy(summary: Mapping[str, Any]) -> None:
    serialized = json.dumps(summary).lower()
    matches = [term for term in PRIVATE_TERMS if term in serialized]
    if matches:
        raise RuntimeError(f"Hidden source fields entered public output: {matches}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_label(source: str) -> str:
    return "Physical IID" if source == "physical_iid" else "One2Six"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "n/a"
    return f"[{value[0]:.4f}, {value[1]:.4f}]"
