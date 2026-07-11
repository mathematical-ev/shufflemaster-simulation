# SPDX-License-Identifier: GPL-3.0-or-later

"""Observable current-rack exclusion and returned-batch response experiment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from experiments.multi_box_counterfactual import (
    CardSourceType,
    SourceName,
    _make_source,
)
from experiments.plots import plot_memory_horizon, plot_response_slopes
from experiments.single_box_game_validation import student_t_summary
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)

OutcomeName = Literal["hi_lo", "low", "neutral", "ten_value", "ace"]
RackScoreBand = Literal["high_heavy", "neutral", "low_heavy"]
ResponseScope = Literal["primary", "position", "exact_lag", "lag_band"]

SOURCE_NAMES: Final[tuple[SourceName, ...]] = ("physical_iid", "one2six")
OUTCOMES: Final[tuple[OutcomeName, ...]] = (
    "hi_lo",
    "low",
    "neutral",
    "ten_value",
    "ace",
)
LOW_RANKS: Final[frozenset[Rank]] = frozenset({"2", "3", "4", "5", "6"})
NEUTRAL_RANKS: Final[frozenset[Rank]] = frozenset({"7", "8", "9"})
TEN_VALUE_RANKS: Final[frozenset[Rank]] = frozenset({"T", "J", "Q", "K"})
BASELINE_PROBABILITIES: Final[dict[str, float]] = {
    "low": 120.0 / 312.0,
    "neutral": 72.0 / 312.0,
    "ten_value": 96.0 / 312.0,
    "ace": 24.0 / 312.0,
}
DEFAULT_LAG_BANDS: Final[tuple[tuple[int, int], ...]] = (
    (1, 15),
    (16, 50),
    (51, 100),
    (101, 250),
    (251, 500),
    (501, 1_000),
)
PRIVATE_EXPORT_TERMS: Final[tuple[str, ...]] = (
    "physical_id",
    "draw_id",
    "shelf_id",
    "buffer_position",
    "feeder",
    "carousel",
    "rng_state",
    "source_snapshot",
)


@dataclass(frozen=True, slots=True)
class ObservableCardResponseConfig:
    """Configuration for immediate and delayed composition response."""

    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    deck_count: int = 6
    base_bet: float = 10.0
    current_rack_states_per_seed: int = 3_000
    current_rack_burn_in_rounds: int = 1_000
    current_rack_sample_interval_rounds: int = 5
    current_rack_probe_cards: int = 15
    lag_rounds_per_seed: int = 50_000
    lag_burn_in_rounds: int = 1_000
    lag_horizon_cards: int = 1_000
    lag_bands: tuple[tuple[int, int], ...] = DEFAULT_LAG_BANDS
    memory_slope_threshold: float = 0.10
    output_dir: Path = Path("experiments/outputs/observable_card_response_5seed")

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed must be supplied.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique.")
        if self.deck_count != 6:
            raise ValueError("observable response requires exactly six decks.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        if self.current_rack_states_per_seed <= 0:
            raise ValueError("current_rack_states_per_seed must be positive.")
        if self.current_rack_burn_in_rounds < 0:
            raise ValueError("current_rack_burn_in_rounds must be non-negative.")
        if self.current_rack_sample_interval_rounds <= 0:
            raise ValueError("current_rack_sample_interval_rounds must be positive.")
        if self.current_rack_probe_cards != 15:
            raise ValueError("current_rack_probe_cards must equal 15.")
        if self.lag_rounds_per_seed <= 0:
            raise ValueError("lag_rounds_per_seed must be positive.")
        if self.lag_burn_in_rounds < 0:
            raise ValueError("lag_burn_in_rounds must be non-negative.")
        if self.lag_horizon_cards < 1_000:
            raise ValueError("lag_horizon_cards must be at least 1,000.")
        if self.memory_slope_threshold <= 0:
            raise ValueError("memory_slope_threshold must be positive.")
        previous_end = 0
        for start, end in self.lag_bands:
            if start <= previous_end or end < start:
                raise ValueError("lag bands must be ordered and non-overlapping.")
            if start < 1 or end > self.lag_horizon_cards:
                raise ValueError("lag bands must lie within the lag horizon.")
            previous_end = end


@dataclass(frozen=True, slots=True)
class CardIndicators:
    """Canonical composition indicators for one card."""

    hi_lo: int
    low: int
    neutral: int
    ten_value: int
    ace: int


@dataclass(frozen=True, slots=True)
class RackFeatures:
    """Observable rack composition and finite-removal benchmarks."""

    rack_size: int
    rack_hi_lo_count: int
    rack_low_count: int
    rack_neutral_count: int
    rack_ten_value_count: int
    rack_ace_count: int
    remaining_card_count: int
    finite_pool_expected_hi_lo: float
    finite_pool_low_probability: float
    finite_pool_neutral_probability: float
    finite_pool_ten_value_probability: float
    finite_pool_ace_probability: float
    finite_pool_low_shift: float
    finite_pool_neutral_shift: float
    finite_pool_ten_value_shift: float
    finite_pool_ace_shift: float
    rack_score_band: RackScoreBand

    def predictor(self, outcome: OutcomeName) -> float:
        """Return the finite-pool predictor for one response outcome."""
        if outcome == "hi_lo":
            return self.finite_pool_expected_hi_lo
        return float(getattr(self, f"finite_pool_{outcome}_shift"))

    def observable_record(self) -> dict[str, Any]:
        """Return only composition-derived public fields."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CurrentRackSample:
    """One immutable current-rack probe and its observable predictors."""

    source: SourceName
    seed: int
    state_index: int
    features: RackFeatures
    probe: tuple[CardIndicators, ...]


@dataclass(frozen=True, slots=True)
class ReturnedBatchEvent:
    """Observable discard batch at its accepted return boundary."""

    batch_id: int
    source: SourceName
    seed: int
    return_draw_index: int
    features: RackFeatures


@dataclass(slots=True)
class RegressionAccumulator:
    """Streaming simple linear regression with an intercept."""

    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0

    def add(self, predictor: float, outcome: float) -> None:
        """Add one predictor/outcome observation."""
        self.count += 1
        self.sum_x += predictor
        self.sum_y += outcome
        self.sum_xx += predictor * predictor
        self.sum_xy += predictor * outcome

    def merge(self, other: RegressionAccumulator) -> None:
        """Merge another sufficient-statistics accumulator."""
        self.count += other.count
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_xx += other.sum_xx
        self.sum_xy += other.sum_xy

    def slope(self) -> float | None:
        """Return the OLS slope, or missing when predictor variance is zero."""
        denominator = self.count * self.sum_xx - self.sum_x * self.sum_x
        if self.count < 2 or abs(denominator) < 1e-18:
            return None
        return (self.count * self.sum_xy - self.sum_x * self.sum_y) / denominator

    def intercept(self) -> float | None:
        """Return the OLS intercept when the slope is defined."""
        slope = self.slope()
        if slope is None:
            return None
        return (self.sum_y - slope * self.sum_x) / self.count


@dataclass(frozen=True, slots=True)
class PrefixCategorySums:
    """Prefix sums for constant-time future composition windows."""

    values: Mapping[OutcomeName, tuple[int, ...]]

    @classmethod
    def from_draws(
        cls, draws: Mapping[OutcomeName, Sequence[int]]
    ) -> PrefixCategorySums:
        """Build one cumulative array per canonical outcome."""
        prefixes: dict[OutcomeName, tuple[int, ...]] = {}
        for outcome in OUTCOMES:
            running = [0]
            for value in draws[outcome]:
                running.append(running[-1] + value)
            prefixes[outcome] = tuple(running)
        return cls(prefixes)

    def window_mean(
        self,
        outcome: OutcomeName,
        *,
        return_draw_index: int,
        start_lag: int,
        end_lag: int,
    ) -> float | None:
        """Return a complete future lag-window mean without off-by-one drift."""
        start_index = return_draw_index + start_lag - 1
        stop_index = return_draw_index + end_lag
        prefix = self.values[outcome]
        if start_index < 0 or stop_index >= len(prefix):
            return None
        total = prefix[stop_index] - prefix[start_index]
        return total / (end_lag - start_lag + 1)


@dataclass(slots=True)
class RecordingCardSource:
    """Experiment wrapper recording draws and observable discard returns."""

    source: CardSourceType
    source_name: SourceName
    seed: int
    enabled: bool = False
    draws: dict[OutcomeName, list[int]] = field(
        default_factory=lambda: {outcome: [] for outcome in OUTCOMES}
    )
    batches: list[ReturnedBatchEvent] = field(default_factory=list)

    @property
    def draw_count(self) -> int:
        """Expose the wrapped source draw count for diagnostics."""
        return self.source.draw_count

    def before_round(self) -> None:
        """Delegate source round-boundary maintenance."""
        self.source.before_round()

    def draw_card(self) -> Card:
        """Draw one card and retain only compact category arrays."""
        card = self.source.draw_card()
        if self.enabled:
            indicators = card_indicators(card.rank)
            for outcome in OUTCOMES:
                self.draws[outcome].append(int(getattr(indicators, outcome)))
        return card

    def accept_discards(self, cards: Sequence[Card]) -> None:
        """Record observable batch composition before delegating its return."""
        if self.enabled:
            self.batches.append(
                ReturnedBatchEvent(
                    batch_id=len(self.batches),
                    source=self.source_name,
                    seed=self.seed,
                    return_draw_index=len(self.draws["hi_lo"]),
                    features=rack_features(cards),
                )
            )
        self.source.accept_discards(cards)


def card_indicators(rank: Rank) -> CardIndicators:
    """Classify one rank into canonical composition categories."""
    low = int(rank in LOW_RANKS)
    neutral = int(rank in NEUTRAL_RANKS)
    ten_value = int(rank in TEN_VALUE_RANKS)
    ace = int(rank == "A")
    return CardIndicators(
        hi_lo=low - ten_value - ace,
        low=low,
        neutral=neutral,
        ten_value=ten_value,
        ace=ace,
    )


def rack_features(cards: Sequence[Card]) -> RackFeatures:
    """Calculate observable counts and six-deck finite-removal benchmarks."""
    indicators = [card_indicators(card.rank) for card in cards]
    rack_size = len(indicators)
    low = sum(item.low for item in indicators)
    neutral = sum(item.neutral for item in indicators)
    ten_value = sum(item.ten_value for item in indicators)
    ace = sum(item.ace for item in indicators)
    hi_lo = sum(item.hi_lo for item in indicators)
    if low + neutral + ten_value + ace != rack_size:
        raise RuntimeError("Rack category counts do not reconcile with rack size.")
    remaining = 312 - rack_size
    if remaining <= 0:
        raise ValueError("Rack must leave cards in the six-deck population.")
    low_probability = (120 - low) / remaining
    neutral_probability = (72 - neutral) / remaining
    ten_probability = (96 - ten_value) / remaining
    ace_probability = (24 - ace) / remaining
    return RackFeatures(
        rack_size=rack_size,
        rack_hi_lo_count=hi_lo,
        rack_low_count=low,
        rack_neutral_count=neutral,
        rack_ten_value_count=ten_value,
        rack_ace_count=ace,
        remaining_card_count=remaining,
        finite_pool_expected_hi_lo=-hi_lo / remaining,
        finite_pool_low_probability=low_probability,
        finite_pool_neutral_probability=neutral_probability,
        finite_pool_ten_value_probability=ten_probability,
        finite_pool_ace_probability=ace_probability,
        finite_pool_low_shift=low_probability - BASELINE_PROBABILITIES["low"],
        finite_pool_neutral_shift=(
            neutral_probability - BASELINE_PROBABILITIES["neutral"]
        ),
        finite_pool_ten_value_shift=(
            ten_probability - BASELINE_PROBABILITIES["ten_value"]
        ),
        finite_pool_ace_shift=ace_probability - BASELINE_PROBABILITIES["ace"],
        rack_score_band=rack_score_band(hi_lo),
    )


def rack_score_band(hi_lo_count: int) -> RackScoreBand:
    """Return the predefined three-level descriptive rack-score band."""
    if hi_lo_count <= -3:
        return "high_heavy"
    if hi_lo_count <= 2:
        return "neutral"
    return "low_heavy"


def initial_deal_position_map(box_count: int) -> dict[str, Any]:
    """Return the valid probe prefix and role positions for a box count."""
    if not 1 <= box_count <= 7:
        raise ValueError("box_count must be between 1 and 7.")
    return {
        "box_count": box_count,
        "valid_probe_prefix": 2 * box_count + 1,
        "player_first_card_positions": list(range(1, box_count + 1)),
        "dealer_upcard_position": box_count + 1,
        "player_second_card_positions": list(range(box_count + 2, 2 * box_count + 2)),
    }


def run_observable_card_response_experiment(
    config: ObservableCardResponseConfig,
) -> dict[str, Any]:
    """Run current-rack probes and delayed returned-batch response analysis."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    current_samples: list[CurrentRackSample] = []
    for source_name in SOURCE_NAMES:
        for seed in config.seeds:
            current_samples.extend(
                _generate_current_rack_samples(config, source_name, seed)
            )

    (
        current_per_seed,
        current_primary,
        current_positions,
    ) = _current_response_rows(config, current_samples)
    role_rows = _current_role_rows(current_samples)
    state_frequency_rows = _current_state_frequency_rows(config, current_samples)

    returned_per_seed: list[dict[str, Any]] = []
    window_count_rows: list[dict[str, Any]] = []
    pooled_lag_accumulators: dict[
        tuple[SourceName, ResponseScope, str, OutcomeName], RegressionAccumulator
    ] = defaultdict(RegressionAccumulator)
    batch_counts: dict[SourceName, int] = defaultdict(int)
    for source_name in SOURCE_NAMES:
        for seed in config.seeds:
            seed_rows, count_rows, accumulators, returned_count = _run_lag_seed(
                config,
                source_name,
                seed,
            )
            returned_per_seed.extend(seed_rows)
            window_count_rows.extend(count_rows)
            batch_counts[source_name] += returned_count
            for key, accumulator in accumulators.items():
                pooled_lag_accumulators[key].merge(accumulator)

    returned_exact = _aggregate_lag_rows(
        returned_per_seed,
        pooled_lag_accumulators,
        scope="exact_lag",
    )
    returned_bands = _aggregate_lag_rows(
        returned_per_seed,
        pooled_lag_accumulators,
        scope="lag_band",
    )
    memory_rows = _memory_horizon_rows(
        returned_bands,
        threshold=config.memory_slope_threshold,
    )
    plot_paths = _write_response_plots(
        config.output_dir,
        current_positions,
        returned_exact,
        returned_bands,
        memory_rows,
        lag_bands=config.lag_bands,
    )

    position_maps = {
        str(box_count): initial_deal_position_map(box_count)
        for box_count in range(1, 8)
    }
    config_payload = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "baseline_category_counts": {
            "low": 120,
            "neutral": 72,
            "ten_value": 96,
            "ace": 24,
            "total": 312,
        },
        "baseline_category_probabilities": BASELINE_PROBABILITIES,
        "position_mapping": position_maps,
        "behavior_policy": {
            "box_count": 1,
            "flat_base_bet": config.base_bet,
            "source_blind_strategy": "PublishedApproxCasinoStrategy",
        },
    }
    summary = {
        "experiment": "observable_card_composition_response",
        "config": config_payload,
        "predictor_fields": [
            "rack_size",
            "rack_hi_lo_count",
            "rack_low_count",
            "rack_neutral_count",
            "rack_ten_value_count",
            "rack_ace_count",
            "remaining_card_count",
            "finite_pool_expected_hi_lo",
            "finite_pool_low_shift",
            "finite_pool_neutral_shift",
            "finite_pool_ten_value_shift",
            "finite_pool_ace_shift",
        ],
        "hidden_state_exported": False,
        "current_rack_primary_response": current_primary,
        "current_rack_position_response": current_positions,
        "current_rack_per_seed_slopes": current_per_seed,
        "current_rack_role_summary": role_rows,
        "current_rack_state_frequency": state_frequency_rows,
        "returned_batch_lag_response": returned_bands,
        "returned_batch_exact_lag_response": returned_exact,
        "returned_batch_per_seed_slopes": returned_per_seed,
        "returned_batch_window_counts": window_count_rows,
        "returned_batch_counts": dict(batch_counts),
        "memory_horizon_summary": memory_rows,
        "plot_paths": plot_paths,
        "interpretation_policy": (
            "Composition responses are observable predictive associations, not a "
            "betting strategy or isolated-batch causal effect."
        ),
    }
    _validate_public_payload(summary)
    _write_json(config.output_dir / "summary.json", summary)
    _write_json(config.output_dir / "experiment_config.json", config_payload)
    _write_csv(config.output_dir / "current_rack_primary_response.csv", current_primary)
    _write_csv(
        config.output_dir / "current_rack_position_response.csv", current_positions
    )
    _write_csv(config.output_dir / "current_rack_per_seed_slopes.csv", current_per_seed)
    _write_csv(config.output_dir / "current_rack_role_summary.csv", role_rows)
    _write_csv(
        config.output_dir / "current_rack_state_frequency.csv",
        state_frequency_rows,
    )
    _write_csv(config.output_dir / "returned_batch_lag_response.csv", returned_bands)
    _write_csv(
        config.output_dir / "returned_batch_exact_lag_response.csv", returned_exact
    )
    _write_csv(
        config.output_dir / "returned_batch_per_seed_slopes.csv", returned_per_seed
    )
    _write_csv(
        config.output_dir / "returned_batch_window_counts.csv", window_count_rows
    )
    _write_csv(config.output_dir / "memory_horizon_summary.csv", memory_rows)
    (config.output_dir / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _generate_current_rack_samples(
    config: ObservableCardResponseConfig,
    source_name: SourceName,
    seed: int,
) -> list[CurrentRackSample]:
    source = _make_source(source_name, config.deck_count, seed)
    game = _behavior_game(config)
    strategy = PublishedApproxCasinoStrategy()
    next_round = 0
    for _ in range(max(1, config.current_rack_burn_in_rounds)):
        game.play_round(
            round_index=next_round,
            card_source=source,
            strategy=strategy,
        )
        next_round += 1

    samples: list[CurrentRackSample] = []
    for state_index in range(config.current_rack_states_per_seed):
        visible_rack = game.pending_discard_rack
        features = rack_features(visible_rack)
        original_draw_count = source.draw_count
        probe_source = deepcopy(source)
        probe_cards = tuple(
            probe_source.draw_card() for _ in range(config.current_rack_probe_cards)
        )
        if source.draw_count != original_draw_count:
            raise RuntimeError("Current-rack probe mutated the behavior trajectory.")
        if game.pending_discard_rack != visible_rack:
            raise RuntimeError(
                "Current-rack probe mutated or returned the visible rack."
            )
        if source_name == "one2six":
            probe_source.assert_invariants([*visible_rack, *probe_cards])
        samples.append(
            CurrentRackSample(
                source=source_name,
                seed=seed,
                state_index=state_index,
                features=features,
                probe=tuple(card_indicators(card.rank) for card in probe_cards),
            )
        )
        if state_index + 1 < config.current_rack_states_per_seed:
            for _ in range(config.current_rack_sample_interval_rounds):
                game.play_round(
                    round_index=next_round,
                    card_source=source,
                    strategy=strategy,
                )
                next_round += 1
    return samples


def _run_lag_seed(
    config: ObservableCardResponseConfig,
    source_name: SourceName,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[SourceName, ResponseScope, str, OutcomeName], RegressionAccumulator],
    int,
]:
    wrapped = RecordingCardSource(
        source=_make_source(source_name, config.deck_count, seed),
        source_name=source_name,
        seed=seed,
    )
    game = _behavior_game(config)
    strategy = PublishedApproxCasinoStrategy()
    next_round = 0
    for _ in range(max(1, config.lag_burn_in_rounds)):
        game.play_round(
            round_index=next_round,
            card_source=wrapped,
            strategy=strategy,
        )
        next_round += 1
    wrapped.enabled = True
    for _ in range(config.lag_rounds_per_seed):
        game.play_round(
            round_index=next_round,
            card_source=wrapped,
            strategy=strategy,
        )
        next_round += 1
    if source_name == "one2six":
        wrapped.source.assert_invariants(game.pending_discard_rack)

    prefixes = PrefixCategorySums.from_draws(wrapped.draws)
    accumulators: dict[
        tuple[SourceName, ResponseScope, str, OutcomeName], RegressionAccumulator
    ] = defaultdict(RegressionAccumulator)
    counts: dict[tuple[ResponseScope, str], list[int]] = defaultdict(lambda: [0, 0])
    for batch in wrapped.batches:
        for exact_lag in range(1, 16):
            label = str(exact_lag)
            usable = _add_lag_observations(
                prefixes,
                batch,
                start_lag=exact_lag,
                end_lag=exact_lag,
                scope="exact_lag",
                label=label,
                accumulators=accumulators,
            )
            counts[("exact_lag", label)][0 if usable else 1] += 1
        for start_lag, end_lag in config.lag_bands:
            label = f"{start_lag}-{end_lag}"
            usable = _add_lag_observations(
                prefixes,
                batch,
                start_lag=start_lag,
                end_lag=end_lag,
                scope="lag_band",
                label=label,
                accumulators=accumulators,
            )
            counts[("lag_band", label)][0 if usable else 1] += 1

    per_seed_rows: list[dict[str, Any]] = []
    for (source, scope, label, outcome), accumulator in sorted(
        accumulators.items(), key=lambda item: str(item[0])
    ):
        per_seed_rows.append(
            {
                "source": source,
                "seed": seed,
                "scope": scope,
                "lag": label,
                "outcome": outcome,
                "usable_batches": accumulator.count,
                "slope": accumulator.slope(),
                "intercept": accumulator.intercept(),
            }
        )
    count_rows = [
        {
            "source": source_name,
            "seed": seed,
            "scope": scope,
            "lag": label,
            "usable_batches": usable_censored[0],
            "censored_batches": usable_censored[1],
            "total_returned_batches": len(wrapped.batches),
        }
        for (scope, label), usable_censored in sorted(
            counts.items(), key=lambda item: str(item[0])
        )
    ]
    return per_seed_rows, count_rows, accumulators, len(wrapped.batches)


def _add_lag_observations(
    prefixes: PrefixCategorySums,
    batch: ReturnedBatchEvent,
    *,
    start_lag: int,
    end_lag: int,
    scope: ResponseScope,
    label: str,
    accumulators: dict[
        tuple[SourceName, ResponseScope, str, OutcomeName], RegressionAccumulator
    ],
) -> bool:
    values = {
        outcome: prefixes.window_mean(
            outcome,
            return_draw_index=batch.return_draw_index,
            start_lag=start_lag,
            end_lag=end_lag,
        )
        for outcome in OUTCOMES
    }
    if any(value is None for value in values.values()):
        return False
    for outcome in OUTCOMES:
        value = values[outcome]
        if value is None:
            raise RuntimeError("Complete lag window unexpectedly became missing.")
        centered = value - BASELINE_PROBABILITIES.get(outcome, 0.0)
        accumulators[(batch.source, scope, label, outcome)].add(
            batch.features.predictor(outcome),
            centered,
        )
    return True


def _current_response_rows(
    config: ObservableCardResponseConfig,
    samples: Sequence[CurrentRackSample],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    pooled: dict[
        tuple[SourceName, ResponseScope, str, OutcomeName], RegressionAccumulator
    ] = defaultdict(RegressionAccumulator)
    for source_name in SOURCE_NAMES:
        for seed in config.seeds:
            seed_samples = [
                sample
                for sample in samples
                if sample.source == source_name and sample.seed == seed
            ]
            for outcome in OUTCOMES:
                accumulator = RegressionAccumulator()
                for sample in seed_samples:
                    value = _probe_mean(sample, outcome)
                    accumulator.add(
                        sample.features.predictor(outcome),
                        value - BASELINE_PROBABILITIES.get(outcome, 0.0),
                    )
                pooled[(source_name, "primary", "next_15", outcome)].merge(accumulator)
                per_seed.append(
                    _seed_slope_row(
                        source_name,
                        seed,
                        "primary",
                        "next_15",
                        outcome,
                        accumulator,
                    )
                )
            for position in range(1, 16):
                for outcome in OUTCOMES:
                    accumulator = RegressionAccumulator()
                    for sample in seed_samples:
                        value = float(getattr(sample.probe[position - 1], outcome))
                        accumulator.add(
                            sample.features.predictor(outcome),
                            value - BASELINE_PROBABILITIES.get(outcome, 0.0),
                        )
                    label = str(position)
                    pooled[(source_name, "position", label, outcome)].merge(accumulator)
                    per_seed.append(
                        _seed_slope_row(
                            source_name,
                            seed,
                            "position",
                            label,
                            outcome,
                            accumulator,
                        )
                    )
    primary = _aggregate_response_rows(per_seed, pooled, scope="primary")
    positions = _aggregate_response_rows(per_seed, pooled, scope="position")
    return per_seed, primary, positions


def _seed_slope_row(
    source: SourceName,
    seed: int,
    scope: ResponseScope,
    label: str,
    outcome: OutcomeName,
    accumulator: RegressionAccumulator,
) -> dict[str, Any]:
    return {
        "source": source,
        "seed": seed,
        "scope": scope,
        "position_or_lag": label,
        "outcome": outcome,
        "observations": accumulator.count,
        "slope": accumulator.slope(),
        "intercept": accumulator.intercept(),
    }


def _aggregate_response_rows(
    per_seed_rows: Sequence[Mapping[str, Any]],
    pooled: Mapping[
        tuple[SourceName, ResponseScope, str, OutcomeName], RegressionAccumulator
    ],
    *,
    scope: ResponseScope,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(
        (key for key in pooled if key[1] == scope),
        key=lambda key: (key[0], _numeric_label_key(key[2]), key[3]),
    )
    for source, _, label, outcome in keys:
        seed_rows = [
            row
            for row in per_seed_rows
            if row["source"] == source
            and row["scope"] == scope
            and row.get("position_or_lag", row.get("lag")) == label
            and row["outcome"] == outcome
            and row["slope"] is not None
        ]
        slopes = [float(row["slope"]) for row in seed_rows]
        uncertainty = student_t_summary(slopes)
        rows.append(
            {
                "source": source,
                "scope": scope,
                "position_or_lag": label,
                "future_position": int(label) if scope == "position" else None,
                "outcome": outcome,
                "pooled_observations": pooled[(source, scope, label, outcome)].count,
                "pooled_descriptive_slope": pooled[
                    (source, scope, label, outcome)
                ].slope(),
                **_seed_summary_fields(uncertainty, slopes),
            }
        )
    return rows


def _aggregate_lag_rows(
    per_seed_rows: Sequence[Mapping[str, Any]],
    pooled: Mapping[
        tuple[SourceName, ResponseScope, str, OutcomeName], RegressionAccumulator
    ],
    *,
    scope: ResponseScope,
) -> list[dict[str, Any]]:
    normalized = [
        {
            **row,
            "position_or_lag": row["lag"],
        }
        for row in per_seed_rows
        if row["scope"] == scope
    ]
    rows = _aggregate_response_rows(normalized, pooled, scope=scope)
    for row in rows:
        row["lag"] = row.pop("position_or_lag")
        row.pop("future_position", None)
    return rows


def _seed_summary_fields(
    summary: Mapping[str, Any], slopes: Sequence[float]
) -> dict[str, Any]:
    return {
        "contributing_seeds": summary["independent_seed_runs"],
        "mean_seed_slope": summary["mean"],
        "mean_absolute_seed_slope": (
            sum(abs(value) for value in slopes) / len(slopes) if slopes else None
        ),
        "sample_standard_deviation": summary["sample_standard_deviation"],
        "standard_error": summary["standard_error"],
        "student_t_95_ci": summary["student_t_95_ci"],
        "minimum_seed_slope": summary["minimum"],
        "maximum_seed_slope": summary["maximum"],
        "positive_seed_slopes": sum(value > 0 for value in slopes),
        "negative_seed_slopes": sum(value < 0 for value in slopes),
        "zero_seed_slopes": sum(value == 0 for value in slopes),
    }


def _current_role_rows(
    samples: Sequence[CurrentRackSample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        seeds = sorted(
            {sample.seed for sample in samples if sample.source == source_name}
        )
        for seed_or_aggregate in (*seeds, "aggregate"):
            source_samples = [
                sample
                for sample in samples
                if sample.source == source_name
                and (
                    seed_or_aggregate == "aggregate" or sample.seed == seed_or_aggregate
                )
            ]
            for box_count in range(1, 8):
                mapping = initial_deal_position_map(box_count)
                player_positions = [
                    *mapping["player_first_card_positions"],
                    *mapping["player_second_card_positions"],
                ]
                dealer_position = mapping["dealer_upcard_position"]
                for role, positions in (
                    ("player_initial_cards", player_positions),
                    ("dealer_upcard", [dealer_position]),
                ):
                    for band in ("all", "high_heavy", "neutral", "low_heavy"):
                        band_samples = [
                            sample
                            for sample in source_samples
                            if band == "all" or sample.features.rack_score_band == band
                        ]
                        row: dict[str, Any] = {
                            "source": source_name,
                            "seed_or_aggregate": seed_or_aggregate,
                            "box_count": box_count,
                            "role": role,
                            "rack_score_band": band,
                            "sampled_states": len(band_samples),
                            "card_observations": len(band_samples) * len(positions),
                        }
                        for outcome in OUTCOMES:
                            values = [
                                float(getattr(sample.probe[position - 1], outcome))
                                for sample in band_samples
                                for position in positions
                            ]
                            row[f"mean_{outcome}"] = _mean(values)
                            accumulator = RegressionAccumulator()
                            for sample in band_samples:
                                role_mean = _mean(
                                    [
                                        float(
                                            getattr(sample.probe[position - 1], outcome)
                                        )
                                        for position in positions
                                    ]
                                )
                                if role_mean is not None:
                                    accumulator.add(
                                        sample.features.predictor(outcome),
                                        role_mean
                                        - BASELINE_PROBABILITIES.get(outcome, 0.0),
                                    )
                            row[f"response_slope_{outcome}"] = accumulator.slope()
                        rows.append(row)
    return rows


def _current_state_frequency_rows(
    config: ObservableCardResponseConfig,
    samples: Sequence[CurrentRackSample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        for seed_or_aggregate in (*config.seeds, "aggregate"):
            relevant = [
                sample
                for sample in samples
                if sample.source == source_name
                and (
                    seed_or_aggregate == "aggregate" or sample.seed == seed_or_aggregate
                )
            ]
            for band in ("high_heavy", "neutral", "low_heavy"):
                band_samples = [
                    sample
                    for sample in relevant
                    if sample.features.rack_score_band == band
                ]
                rows.append(
                    {
                        "source": source_name,
                        "seed_or_aggregate": seed_or_aggregate,
                        "rack_score_band": band,
                        "state_count": len(band_samples),
                        "state_proportion": (
                            len(band_samples) / len(relevant) if relevant else None
                        ),
                        "mean_rack_size": _mean(
                            [sample.features.rack_size for sample in band_samples]
                        ),
                        "mean_rack_hi_lo_count": _mean(
                            [
                                sample.features.rack_hi_lo_count
                                for sample in band_samples
                            ]
                        ),
                    }
                )
    return rows


def _memory_horizon_rows(
    lag_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        for outcome in OUTCOMES:
            series = [
                row
                for row in lag_rows
                if row["source"] == source_name and row["outcome"] == outcome
            ]
            series.sort(key=lambda row: _numeric_label_key(str(row["lag"])))
            negligible = [
                _is_negligible_response(row, threshold=threshold) for row in series
            ]
            earliest = None
            for index, row in enumerate(series):
                if all(negligible[index:]):
                    earliest = row["lag"]
                    break
            detected = [
                row
                for row in series
                if row["mean_seed_slope"] is not None
                and abs(row["mean_seed_slope"]) >= threshold
                and _interval_excludes_zero(row["student_t_95_ci"])
            ]
            signs = [
                1 if row["mean_seed_slope"] > 0 else -1
                for row in series
                if row["mean_seed_slope"] not in {None, 0}
            ]
            rows.append(
                {
                    "source": source_name,
                    "outcome": outcome,
                    "slope_threshold": threshold,
                    "earliest_lag_all_later_negligible": earliest,
                    "latest_lag_clearly_detected": (
                        detected[-1]["lag"] if detected else None
                    ),
                    "response_changes_sign": any(
                        left != right
                        for left, right in zip(signs, signs[1:], strict=False)
                    ),
                    "contributing_seeds": max(
                        (row["contributing_seeds"] for row in series), default=0
                    ),
                    "interpretation": (
                        "Observable predictive influence of this cohort is no longer "
                        "detected at the current resolution."
                        if earliest is not None
                        else "No all-later negligible horizon was detected."
                    ),
                }
            )
    return rows


def _is_negligible_response(row: Mapping[str, Any], *, threshold: float) -> bool:
    mean_absolute = row["mean_absolute_seed_slope"]
    interval = row["student_t_95_ci"]
    return bool(
        mean_absolute is not None
        and mean_absolute < threshold
        and _interval_contains_zero(interval)
    )


def _write_response_plots(
    output_dir: Path,
    position_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    band_rows: Sequence[Mapping[str, Any]],
    memory_rows: Sequence[Mapping[str, Any]],
    *,
    lag_bands: Sequence[tuple[int, int]],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    plot_outcomes: tuple[OutcomeName, ...] = ("hi_lo", "ten_value", "ace", "low")
    for outcome in plot_outcomes:
        current_name = f"current_rack_{outcome}_response_by_position.png"
        plot_response_slopes(
            [row for row in position_rows if row["outcome"] == outcome],
            x_field="future_position",
            x_values=list(range(1, 16)),
            title=f"Current-rack {outcome.replace('_', ' ')} response by position",
            x_label="Future card position",
            output_path=output_dir / current_name,
            reference_one=True,
        )
        paths[f"current_rack_{outcome}_response_by_position"] = current_name

        returned_name = f"returned_batch_{outcome}_response_by_lag.png"
        combined_rows = [
            *[row for row in exact_rows if row["outcome"] == outcome],
            *[row for row in band_rows if row["outcome"] == outcome],
        ]
        labels = [str(value) for value in range(1, 16)] + [
            f"{start}-{end}" for start, end in lag_bands
        ]
        plot_response_slopes(
            combined_rows,
            x_field="lag",
            x_values=labels,
            title=f"Returned-batch {outcome.replace('_', ' ')} response by lag",
            x_label="Exact future lag, then lag band",
            output_path=output_dir / returned_name,
            reference_one=True,
        )
        paths[f"returned_batch_{outcome}_response_by_lag"] = returned_name

    memory_name = "memory_horizon_overview.png"
    plot_memory_horizon(
        band_rows,
        memory_rows=memory_rows,
        lag_labels=[f"{start}-{end}" for start, end in lag_bands],
        output_path=output_dir / memory_name,
    )
    paths["memory_horizon_overview"] = memory_name
    return paths


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lag_labels = [f"{start}-{end}" for start, end in summary["config"]["lag_bands"]]
    lines = [
        "# Observable Card-Composition Response",
        "",
        "This experiment tests whether player-observable discard composition",
        "predicts future card composition.",
        "",
        "It separates:",
        "",
        "1. immediate exclusion while the visible rack remains outside the source;",
        "2. delayed response after a rack has been returned.",
        "",
        "It does not implement or validate a betting strategy.",
        "",
        "## Current-Rack Primary Response",
        "",
        "| Source | Outcome | Mean slope | 95% seed CI | Interpretation |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary["current_rack_primary_response"]:
        lines.append(
            f"| {_source_label(row['source'])} | {row['outcome']} | "
            f"{_format_number(row['mean_seed_slope'])} | "
            f"{_format_interval(row['student_t_95_ci'])} | "
            f"{_slope_interpretation(row['mean_seed_slope'], row['student_t_95_ci'])} |"
        )
    lines.extend(["", "## Current-Rack Position Response", ""])
    for outcome in OUTCOMES:
        lines.extend(
            [
                f"### {outcome.replace('_', ' ').title()}",
                "",
                "| Source | "
                + " | ".join(str(position) for position in range(1, 16))
                + " |",
                "|---|" + "---:|" * 15,
            ]
        )
        for source_name in SOURCE_NAMES:
            source_rows = [
                row
                for row in summary["current_rack_position_response"]
                if row["source"] == source_name and row["outcome"] == outcome
            ]
            source_rows.sort(key=lambda row: row["future_position"])
            lines.append(
                f"| {_source_label(source_name)} | "
                + " | ".join(
                    _format_number(row["mean_seed_slope"]) for row in source_rows
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Delayed Lag Response",
            "",
            "| Source | Lag | Hi-Lo slope | Low slope | Ten-value slope | Ace slope |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source_name in SOURCE_NAMES:
        for lag in lag_labels:
            values = {
                row["outcome"]: row["mean_seed_slope"]
                for row in summary["returned_batch_lag_response"]
                if row["source"] == source_name and row["lag"] == lag
            }
            lines.append(
                f"| {_source_label(source_name)} | {lag} | "
                f"{_format_number(values['hi_lo'])} | "
                f"{_format_number(values['low'])} | "
                f"{_format_number(values['ten_value'])} | "
                f"{_format_number(values['ace'])} |"
            )

    lines.extend(["", "## Memory Horizon", ""])
    for row in summary["memory_horizon_summary"]:
        lines.append(
            f"- {_source_label(row['source'])} {row['outcome']}: earliest all-later "
            f"negligible lag={row['earliest_lag_all_later_negligible']}; latest "
            f"clearly detected lag={row['latest_lag_clearly_detected']}; sign "
            f"change={row['response_changes_sign']}."
        )
    lines.extend(
        [
            "",
            "This horizon means only that observable predictive influence of the "
            "cohort is no longer detected at the current resolution. It does not "
            "claim that the source is globally IID.",
            "",
            "## Negative-Control Check",
            "",
            _negative_control_text(summary),
            "",
            "## Ace-Specific Note",
            "",
            _ace_role_text(summary),
            "",
            "Ace availability is retained separately for player initial-card and "
            "dealer-upcard positions. Its value is position-dependent and requires "
            "later EV weighting.",
            "",
            "## Streak Note",
            "",
            "Previous unconditional monetary streak testing found no material",
            "difference between Physical IID and One2Six.",
            "",
            "Monetary streaks are therefore not used as predictors in this",
            "experiment.",
            "",
            "## Plots",
            "",
        ]
    )
    for label, path in summary["plot_paths"].items():
        lines.append(f"- [{label}]({path})")
    lines.append("")
    return "\n".join(lines)


def _negative_control_text(summary: Mapping[str, Any]) -> str:
    rows = [
        row
        for row in [
            *summary["current_rack_primary_response"],
            *summary["returned_batch_lag_response"],
        ]
        if row["source"] == "physical_iid"
        and _interval_excludes_zero(row["student_t_95_ci"])
    ]
    if rows:
        return (
            f"Physical IID has {len(rows)} primary or lag-band intervals excluding "
            "zero. Treat this as possible sampling noise, leakage, bias, or analysis "
            "error; do not proceed to strategy construction."
        )
    return "Physical IID primary and lag-band responses are consistent with zero."


def _ace_role_text(summary: Mapping[str, Any]) -> str:
    rows = [
        row
        for row in summary["current_rack_role_summary"]
        if row["source"] == "one2six"
        and row["seed_or_aggregate"] == "aggregate"
        and row["rack_score_band"] == "all"
        and row["box_count"] in {1, 7}
    ]
    return " ".join(
        f"{row['box_count']}-box {row['role']} ace response slope="
        f"{_format_number(row['response_slope_ace'])}."
        for row in rows
    )


def _behavior_game(config: ObservableCardResponseConfig) -> CasinoBlackjackGame:
    return CasinoBlackjackGame(
        CasinoBlackjackConfig(
            base_bet=config.base_bet,
            box_count=1,
            box_bets={1: config.base_bet},
            deck_count=config.deck_count,
        )
    )


def _probe_mean(sample: CurrentRackSample, outcome: OutcomeName) -> float:
    return sum(float(getattr(card, outcome)) for card in sample.probe) / len(
        sample.probe
    )


def _numeric_label_key(label: str) -> tuple[int, int]:
    if "-" in label:
        start, end = label.split("-", maxsplit=1)
        return int(start), int(end)
    value = int(label) if label.isdigit() else 0
    return value, value


def _interval_contains_zero(interval: object) -> bool:
    return (
        isinstance(interval, list)
        and len(interval) == 2
        and interval[0] <= 0 <= interval[1]
    )


def _interval_excludes_zero(interval: object) -> bool:
    return (
        isinstance(interval, list)
        and len(interval) == 2
        and (interval[0] > 0 or interval[1] < 0)
    )


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _validate_public_payload(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload).lower()
    matches = [term for term in PRIVATE_EXPORT_TERMS if term in serialized]
    if matches:
        raise RuntimeError(f"Hidden-state terms entered public output: {matches}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_label(source_name: str) -> str:
    return "Physical IID" if source_name == "physical_iid" else "One2Six"


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_interval(interval: object) -> str:
    if not isinstance(interval, list) or len(interval) != 2:
        return "n/a"
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _slope_interpretation(value: float | None, interval: object) -> str:
    if value is None:
        return "missing predictor variance"
    if _interval_contains_zero(interval):
        return "uncertain; seed CI includes zero"
    if abs(value) < 0.10:
        return "near zero"
    if value < 0:
        return "reversal"
    if value < 0.75:
        return "partial finite-removal response"
    if value <= 1.25:
        return "near direct finite-pool response"
    return "amplified positive response"
