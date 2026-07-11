# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare one-box blackjack play under physical IID and One2Six sources."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Final, Literal

from experiments.metrics import (
    MonetaryStreakTracker,
    streak_frequency_summary,
    validate_streak_reconciliation,
)
from experiments.plots import plot_comparative_signed_streak_histogram
from shufflemaster_sim.actions import ActionType, GameAction
from shufflemaster_sim.card_sources import (
    One2SixCardSource,
    One2SixConfig,
    PhysicalIidCardSource,
)
from shufflemaster_sim.cards import Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
    CasinoBlackjackStrategy,
)
from shufflemaster_sim.hand_values import is_natural_blackjack, split_value_from_rank
from shufflemaster_sim.results import ResultRecorder, SimulationResult
from shufflemaster_sim.state import BlackjackDecisionState, TableState
from shufflemaster_sim.strategies.published_casino_strategy import (
    DEALER_UPCARDS,
    PAIR_TOTALS,
    PublishedApproxCasinoStrategy,
)

SourceName = Literal["physical_iid", "one2six"]
SourceRuns = dict[str, list[dict[str, Any]]]
SOURCE_NAMES: Final[tuple[SourceName, ...]] = ("physical_iid", "one2six")
PAIR_CATEGORIES: Final[tuple[str, ...]] = (
    "A-A",
    "2-2",
    "3-3",
    "4-4",
    "5-5",
    "6-6",
    "7-7",
    "8-8",
    "9-9",
    "ten-value pair",
)
UPCARD_CATEGORIES: Final[tuple[str, ...]] = DEALER_UPCARDS
NON_SPLIT_ACTIONS: Final[tuple[str, ...]] = ("hit", "stand", "double")
PHYSICAL_IID_PAIR_OPPORTUNITY_RATE: Final[float] = 25.0 / 169.0
NORMAL_95_CRITICAL: Final[float] = 1.959963984540054
VALIDATION_NOTE = (
    "This is a long-run game-engine and source-integration validation experiment. "
    "It is not a state-dependent advantage-play test."
)

# Two-sided 95% Student-t critical values. For unsupported df above 30, the
# largest tabulated df not exceeding the requested df is used conservatively.
_T_CRITICAL_95: Final[dict[int, float]] = {
    1: 12.706204736,
    2: 4.30265273,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.20098516,
    12: 2.17881283,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.10092204,
    19: 2.093024054,
    20: 2.085963447,
    21: 2.079613845,
    22: 2.073873068,
    23: 2.06865761,
    24: 2.063898562,
    25: 2.059538553,
    26: 2.055529439,
    27: 2.051830516,
    28: 2.048407142,
    29: 2.045229642,
    30: 2.042272456,
    40: 2.02107539,
    60: 2.00029782,
    120: 1.9799304,
}


@dataclass(frozen=True, slots=True)
class SingleBoxGameValidationConfig:
    """Configuration for a flat-bet, six-deck, one-box comparison."""

    rounds: int = 1_000_000
    base_bet: float = 10.0
    seeds: tuple[int, ...] = (42,)
    deck_count: int = 6
    output_dir: Path = Path("experiments/outputs/single_box_game_validation_1m")
    include_pair_upcard_detail: bool = True

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        if self.deck_count != 6:
            raise ValueError("single-box game validation requires exactly six decks.")
        if not self.seeds:
            raise ValueError("at least one seed must be supplied.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique independent run identifiers.")


@dataclass(slots=True)
class RunningMoments:
    """Streaming Welford accumulator for round-level player profit."""

    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    _m2: float = 0.0

    def add(self, value: float) -> None:
        """Add one observation without retaining it."""
        self.count += 1
        self.total += value
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    def as_dict(self) -> dict[str, Any]:
        """Return descriptive round-level moments and a naive normal interval."""
        variance = self._m2 / (self.count - 1) if self.count > 1 else None
        standard_deviation = sqrt(variance) if variance is not None else None
        standard_error = (
            standard_deviation / sqrt(self.count)
            if standard_deviation is not None
            else None
        )
        confidence_interval = (
            [
                self.mean - NORMAL_95_CRITICAL * standard_error,
                self.mean + NORMAL_95_CRITICAL * standard_error,
            ]
            if standard_error is not None
            else None
        )
        return {
            "count": self.count,
            "sum": self.total,
            "mean_round_profit": self.mean if self.count else None,
            "sample_variance_round_profit": variance,
            "sample_standard_deviation_round_profit": standard_deviation,
            "naive_round_se": standard_error,
            "naive_normal_95_ci": confidence_interval,
            "warning": (
                "Descriptive only; serial dependence can invalidate round-IID "
                "confidence interpretation."
            ),
        }


@dataclass(slots=True)
class _RunCounters:
    dealer_blackjacks: int = 0
    total_original_hands: int = 0
    total_final_player_hands: int = 0

    def observe_round(self, table: TableState) -> None:
        self.dealer_blackjacks += int(is_natural_blackjack(table.dealer.cards))
        self.total_original_hands += len(table.boxes)
        self.total_final_player_hands += sum(len(box.hands) for box in table.boxes)


@dataclass(slots=True)
class PairActionRecorder:
    """Source-blind strategy wrapper that decomposes initial pair decisions."""

    strategy: CasinoBlackjackStrategy
    initial_pair_opportunities: int = 0
    actual_split_actions: int = 0
    initial_pairs_not_split: int = 0
    expected_split_missing_legal_action: int = 0
    actual_split_without_initial_pair: int = 0
    non_split_actions: dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in NON_SPLIT_ACTIONS}
    )
    categories: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            category: _empty_pair_cell(include_upcards=True)
            for category in PAIR_CATEGORIES
        }
    )

    def choose_action(self, *, decision: BlackjackDecisionState) -> GameAction:
        """Delegate the decision unchanged and record initial-pair decomposition."""
        action = self.strategy.choose_action(decision=decision)
        category = initial_pair_category(
            decision.player_ranks,
            is_split_hand=decision.is_split_hand,
        )
        if action.action_type == ActionType.SPLIT:
            self.actual_split_actions += 1
            if category is None:
                self.actual_split_without_initial_pair += 1

        if category is None:
            return action

        self.initial_pair_opportunities += 1
        upcard = dealer_upcard_category(decision.dealer_upcard_rank)
        cell = self.categories[category]
        upcard_cell = cell["by_upcard"][upcard]
        cell["opportunities"] += 1
        upcard_cell["opportunities"] += 1

        if _published_pair_expects_split(category, upcard) and (
            ActionType.SPLIT not in decision.legal_actions
        ):
            self.expected_split_missing_legal_action += 1

        if action.action_type == ActionType.SPLIT:
            cell["actual_splits"] += 1
            upcard_cell["actual_splits"] += 1
            return action

        action_name = action.action_type.value
        if action_name not in self.non_split_actions:
            raise RuntimeError(f"Unexpected non-split pair action: {action_name}.")
        self.initial_pairs_not_split += 1
        self.non_split_actions[action_name] += 1
        cell["non_split_actions"][action_name] += 1
        upcard_cell["non_split_actions"][action_name] += 1
        return action

    def as_metrics(self, rounds: int) -> dict[str, Any]:
        """Return reconciled pair diagnostics with rates."""
        categories = {
            category: _pair_cell_metrics(cell, rounds)
            for category, cell in self.categories.items()
        }
        metrics = {
            "total_original_rounds": rounds,
            "initial_pair_opportunities": self.initial_pair_opportunities,
            "initial_pair_opportunity_rate": _rate(
                self.initial_pair_opportunities,
                rounds,
            ),
            "actual_split_actions": self.actual_split_actions,
            "actual_split_rate_per_round": _rate(self.actual_split_actions, rounds),
            "split_rate_given_initial_pair": _rate(
                self.actual_split_actions,
                self.initial_pair_opportunities,
            ),
            "initial_pairs_not_split": self.initial_pairs_not_split,
            "non_split_actions": dict(self.non_split_actions),
            "expected_split_missing_legal_action": (
                self.expected_split_missing_legal_action
            ),
            "actual_split_without_initial_pair": (
                self.actual_split_without_initial_pair
            ),
            "categories": categories,
        }
        validate_pair_reconciliation(metrics)
        return metrics


def initial_pair_category(
    player_ranks: tuple[Rank, ...],
    *,
    is_split_hand: bool,
) -> str | None:
    """Return the original equal-value pair category, if any."""
    if is_split_hand or len(player_ranks) != 2:
        return None
    first_value = split_value_from_rank(player_ranks[0])
    if first_value != split_value_from_rank(player_ranks[1]):
        return None
    if first_value == "A":
        return "A-A"
    if first_value == 10:
        return "ten-value pair"
    return f"{first_value}-{first_value}"


def dealer_upcard_category(rank: Rank) -> str:
    """Return the strategy-table dealer upcard category."""
    value = split_value_from_rank(rank)
    return "T" if value == 10 else str(value)


def validate_pair_reconciliation(metrics: dict[str, Any]) -> None:
    """Raise if pair opportunity and action decompositions do not reconcile."""
    opportunities = metrics["initial_pair_opportunities"]
    category_opportunities = sum(
        cell["opportunities"] for cell in metrics["categories"].values()
    )
    upcard_opportunities = sum(
        upcard_cell["opportunities"]
        for cell in metrics["categories"].values()
        for upcard_cell in cell["by_upcard"].values()
    )
    non_split_total = sum(metrics["non_split_actions"].values())
    if category_opportunities != opportunities:
        raise RuntimeError("Pair categories do not reconcile with opportunities.")
    if upcard_opportunities != opportunities:
        raise RuntimeError("Pair/upcard cells do not reconcile with opportunities.")
    if metrics["actual_split_actions"] > opportunities:
        raise RuntimeError("Actual splits exceed initial pair opportunities.")
    if metrics["actual_split_actions"] + non_split_total != opportunities:
        raise RuntimeError("Split and non-split pair actions do not reconcile.")


def student_t_summary(values: list[float]) -> dict[str, Any]:
    """Summarize independent seed estimates with a Student-t interval."""
    count = len(values)
    mean = sum(values) / count if count else None
    variance = (
        sum((value - mean) ** 2 for value in values) / (count - 1)
        if count > 1 and mean is not None
        else None
    )
    standard_deviation = sqrt(variance) if variance is not None else None
    standard_error = (
        standard_deviation / sqrt(count) if standard_deviation is not None else None
    )
    critical, critical_df = student_t_critical_95(count)
    confidence_interval = (
        [mean - critical * standard_error, mean + critical * standard_error]
        if mean is not None and critical is not None and standard_error is not None
        else None
    )
    return {
        "independent_seed_runs": count,
        "mean": mean,
        "sample_variance": variance,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "student_t_95_ci": confidence_interval,
        "student_t_critical": critical,
        "degrees_of_freedom": count - 1 if count > 0 else None,
        "critical_value_table_df": critical_df,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def student_t_critical_95(sample_count: int) -> tuple[float | None, int | None]:
    """Return a two-sided 95% t critical value and lookup-table df."""
    if sample_count < 2:
        return None, None
    degrees_of_freedom = sample_count - 1
    if degrees_of_freedom > 120:
        return NORMAL_95_CRITICAL, None
    lookup_df = max(df for df in _T_CRITICAL_95 if df <= degrees_of_freedom)
    return _T_CRITICAL_95[lookup_df], lookup_df


def paired_difference_summary(
    physical_runs: list[dict[str, Any]],
    one2six_runs: list[dict[str, Any]],
    field_name: str,
) -> dict[str, Any]:
    """Return paired One2Six-minus-IID seed differences for one field."""
    physical_by_seed = {run["seed"]: run[field_name] for run in physical_runs}
    one2six_by_seed = {run["seed"]: run[field_name] for run in one2six_runs}
    if set(physical_by_seed) != set(one2six_by_seed):
        raise ValueError("Sources must use the same seed list for paired differences.")
    differences = [
        one2six_by_seed[seed] - physical_by_seed[seed]
        for seed in sorted(physical_by_seed)
    ]
    summary = student_t_summary(differences)
    summary.update(
        {
            "differences_by_seed": {
                str(seed): one2six_by_seed[seed] - physical_by_seed[seed]
                for seed in sorted(physical_by_seed)
            },
            "positive_differences": sum(value > 0 for value in differences),
            "negative_differences": sum(value < 0 for value in differences),
            "zero_differences": sum(value == 0 for value in differences),
        }
    )
    return summary


def split_count_decomposition(
    physical: dict[str, Any],
    one2six: dict[str, Any],
) -> dict[str, float | int]:
    """Decompose the split-count gap into pair volume and conditional mix."""
    opportunity_difference = (
        one2six["initial_pair_opportunities"] - physical["initial_pair_opportunities"]
    )
    split_difference = one2six["split_actions"] - physical["split_actions"]
    physical_conditional_rate = physical["split_rate_given_initial_pair"]
    conditional_rate_difference = (
        one2six["split_rate_given_initial_pair"] - physical_conditional_rate
    )
    opportunity_component = opportunity_difference * physical_conditional_rate
    conditional_component = (
        one2six["initial_pair_opportunities"] * conditional_rate_difference
    )
    return {
        "initial_pair_opportunity_count_difference": opportunity_difference,
        "actual_split_count_difference": split_difference,
        "split_given_pair_rate_difference": conditional_rate_difference,
        "opportunity_volume_component_at_iid_conditional_rate": (opportunity_component),
        "conditional_mix_component_at_one2six_pair_count": conditional_component,
        "component_sum": opportunity_component + conditional_component,
    }


def box_round_net_result(table: TableState) -> float:
    """Return the total monetary result of the sole active box."""
    if len(table.boxes) != 1:
        raise ValueError("Single-box validation requires exactly one active box.")
    return sum(hand.net_result for hand in table.boxes[0].hands)


def run_single_box_game_validation(
    config: SingleBoxGameValidationConfig,
    *,
    strategy: CasinoBlackjackStrategy | None = None,
) -> dict[str, Any]:
    """Run both sources for every seed and write aggregate comparison files."""
    fixed_strategy = (
        strategy if strategy is not None else PublishedApproxCasinoStrategy()
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for seed in config.seeds:
        for source_name in SOURCE_NAMES:
            metrics = run_source_game_validation(
                config=config,
                source_name=source_name,
                seed=seed,
                strategy=fixed_strategy,
            )
            runs.append(metrics)
            _write_json(
                config.output_dir / f"{source_name}_seed_{seed}.json",
                metrics,
            )

    runs_by_source = {
        source_name: [run for run in runs if run["source"] == source_name]
        for source_name in SOURCE_NAMES
    }
    aggregates = {
        source_name: _aggregate_source_runs(source_runs)
        for source_name, source_runs in runs_by_source.items()
    }
    seed_uncertainty = _seed_level_uncertainty(runs_by_source)
    split_decomposition = split_count_decomposition(
        aggregates["physical_iid"],
        aggregates["one2six"],
    )
    summary = {
        "experiment": "single_box_game_validation",
        "purpose": VALIDATION_NOTE,
        "config": {
            "rounds_per_seed": config.rounds,
            "base_bet": config.base_bet,
            "seeds": list(config.seeds),
            "deck_count": config.deck_count,
            "box_count": 1,
            "flat_betting": True,
            "dealer_rule": "stand_on_all_17",
            "initial_burn": True,
            "strategy_implementation": type(fixed_strategy).__name__,
            "include_pair_upcard_detail": config.include_pair_upcard_detail,
        },
        "theory": {
            "physical_iid_initial_pair_opportunity_rate": (
                PHYSICAL_IID_PAIR_OPPORTUNITY_RATE
            ),
            "derivation": "9 * (1 / 13)^2 + (4 / 13)^2 = 25 / 169",
        },
        "runs": runs,
        "aggregate": aggregates,
        "seed_level_uncertainty": seed_uncertainty,
        "split_count_decomposition": split_decomposition,
        "streak_source_differences": _streak_source_differences(aggregates),
        "difference_definition": (
            "One2Six minus physical IID; rate differences are percentage points "
            "only when rendered for display."
        ),
        "metric_definitions": _metric_definitions(),
    }
    plot_path = config.output_dir / "signed_streak_length_histogram.png"
    plot_comparative_signed_streak_histogram(
        {
            source_name: aggregates[source_name]["streak_metrics"]
            for source_name in SOURCE_NAMES
        },
        output_path=plot_path,
    )
    summary["plot_paths"] = {
        "signed_streak_length_histogram": plot_path.name,
    }
    _write_json(config.output_dir / "summary.json", summary)
    _write_summary_csv(config.output_dir / "summary.csv", aggregates)
    _write_per_seed_csv(config.output_dir / "per_seed_summary.csv", runs)
    _write_pair_summary_csv(config.output_dir / "pair_summary.csv", aggregates)
    if config.include_pair_upcard_detail:
        _write_pair_upcard_csv(config.output_dir / "pair_by_upcard.csv", aggregates)
    _write_streak_summary_csv(
        config.output_dir / "streak_summary.csv",
        runs,
        aggregates,
    )
    _write_streak_distribution_csv(
        config.output_dir / "streak_length_distribution.csv",
        runs,
        aggregates,
    )
    (config.output_dir / "summary.md").write_text(
        _summary_markdown(config, aggregates, seed_uncertainty),
        encoding="utf-8",
    )
    return summary


def run_source_game_validation(
    *,
    config: SingleBoxGameValidationConfig,
    source_name: SourceName,
    seed: int,
    strategy: CasinoBlackjackStrategy,
) -> dict[str, Any]:
    """Run one card source with the shared rule engine and fixed strategy."""
    source = _make_source(source_name, deck_count=config.deck_count, seed=seed)
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            base_bet=config.base_bet,
            box_count=1,
            box_bets={1: config.base_bet},
            deck_count=config.deck_count,
            dealer_hits_soft_17=False,
            allow_resplit=False,
            max_hands_per_box=2,
            use_shuffling_device=True,
            burn_initial_card=True,
        )
    )
    recorder = ResultRecorder(
        base_bet=config.base_bet,
        box_count=1,
        retain_round_results=False,
    )
    pair_recorder = PairActionRecorder(strategy)
    round_profit = RunningMoments()
    streak_tracker = MonetaryStreakTracker()
    counters = _RunCounters()

    for round_index in range(config.rounds):
        table = game.play_round(
            round_index=round_index,
            card_source=source,
            strategy=pair_recorder,
        )
        round_result = recorder.record_round(table)
        round_profit.add(round_result.net_profit)
        box_net_result = box_round_net_result(table)
        if box_net_result != round_result.net_profit:
            raise RuntimeError("Box net result does not reconcile with round profit.")
        streak_tracker.observe(box_net_result)
        counters.observe_round(table)

    result = recorder.build_result()
    pair_metrics = pair_recorder.as_metrics(result.rounds_played)
    if pair_metrics["actual_split_actions"] != result.box_results[0].splits:
        raise RuntimeError("Instrumented split actions do not match result accounting.")

    invariant_passed: bool | None = None
    invariant_error: str | None = None
    if isinstance(source, One2SixCardSource):
        try:
            source.assert_invariants(external_cards=game.pending_discard_rack)
            invariant_passed = True
        except AssertionError as exc:
            invariant_passed = False
            invariant_error = str(exc)

    return _source_metrics(
        source_name=source_name,
        seed=seed,
        strategy=strategy,
        result=result,
        counters=counters,
        total_cards_drawn=source.draw_count,
        one2six_source=source if isinstance(source, One2SixCardSource) else None,
        invariant_passed=invariant_passed,
        invariant_error=invariant_error,
        pair_metrics=pair_metrics,
        round_profit_statistics=round_profit.as_dict(),
        streak_metrics=streak_tracker.summary(),
    )


def _make_source(
    source_name: SourceName,
    *,
    deck_count: int,
    seed: int,
) -> PhysicalIidCardSource | One2SixCardSource:
    if source_name == "physical_iid":
        return PhysicalIidCardSource(deck_count=deck_count, seed=seed)
    return One2SixCardSource(
        config=One2SixConfig(
            deck_count=deck_count,
            strict_invariants=False,
            retain_event_telemetry=False,
            retain_ejection_records=False,
            retain_accepted_discard_history=False,
        ),
        seed=seed,
    )


def _source_metrics(
    *,
    source_name: SourceName,
    seed: int,
    strategy: CasinoBlackjackStrategy,
    result: SimulationResult,
    counters: _RunCounters,
    total_cards_drawn: int,
    one2six_source: One2SixCardSource | None,
    invariant_passed: bool | None,
    invariant_error: str | None,
    pair_metrics: dict[str, Any],
    round_profit_statistics: dict[str, Any],
    streak_metrics: dict[str, Any],
) -> dict[str, Any]:
    box = result.box_results[0]
    rounds = result.rounds_played
    ejections = one2six_source.ejection_count if one2six_source else None
    fallbacks = one2six_source.fallback_ejection_count if one2six_source else None
    return {
        "source": source_name,
        "seed": seed,
        "strategy_implementation": type(strategy).__name__,
        "deck_count": 6,
        "completed_rounds": rounds,
        "base_bet": box.initial_wagered / rounds if rounds else 0.0,
        "initial_wagered": result.initial_wagered,
        "additional_action_wagered": result.action_wagered,
        "total_wagered": result.total_wagered,
        "net_player_result": result.net_profit,
        "net_casino_result": -result.net_profit,
        "player_net_per_round": _rate(result.net_profit, rounds),
        "player_net_per_initial_wager": result.edge_per_initial_wager,
        "player_net_per_total_wager": result.edge_per_total_wager,
        "round_profit_statistics": round_profit_statistics,
        "streak_metrics": streak_metrics,
        "player_blackjacks": box.blackjacks,
        "player_blackjack_rate": _rate(box.blackjacks, counters.total_original_hands),
        "double_actions": box.doubles,
        "double_actions_per_round": _rate(box.doubles, rounds),
        "split_actions": box.splits,
        "split_actions_per_round": _rate(box.splits, rounds),
        "initial_pair_opportunities": pair_metrics["initial_pair_opportunities"],
        "initial_pair_opportunity_rate": pair_metrics["initial_pair_opportunity_rate"],
        "split_rate_given_initial_pair": pair_metrics["split_rate_given_initial_pair"],
        "pair_diagnostics": pair_metrics,
        "dealer_blackjacks": counters.dealer_blackjacks,
        "dealer_blackjack_rate": _rate(counters.dealer_blackjacks, rounds),
        "total_wins": box.wins,
        "total_losses": box.losses,
        "total_pushes": box.pushes,
        "total_original_hands": counters.total_original_hands,
        "total_final_player_hands": counters.total_final_player_hands,
        "total_cards_drawn": total_cards_drawn,
        "average_cards_drawn_per_round": _rate(total_cards_drawn, rounds),
        "one2six_invariant_passed": invariant_passed,
        "one2six_invariant_error": invariant_error,
        "one2six_final_invariant_check": one2six_source is not None,
        "one2six_accepted_discard_batch_count": (
            one2six_source.accepted_discard_batch_count
            if one2six_source is not None
            else None
        ),
        "one2six_ejection_count": ejections,
        "one2six_fallback_count": fallbacks,
        "one2six_fallback_rate": (
            _rate(fallbacks, ejections)
            if fallbacks is not None and ejections is not None
            else None
        ),
    }


def _aggregate_source_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    additive_fields = (
        "completed_rounds",
        "initial_wagered",
        "additional_action_wagered",
        "total_wagered",
        "net_player_result",
        "net_casino_result",
        "player_blackjacks",
        "double_actions",
        "split_actions",
        "initial_pair_opportunities",
        "dealer_blackjacks",
        "total_wins",
        "total_losses",
        "total_pushes",
        "total_original_hands",
        "total_final_player_hands",
        "total_cards_drawn",
    )
    aggregate = {field: sum(run[field] for run in runs) for field in additive_fields}
    aggregate.update(
        {
            "source": runs[0]["source"],
            "seeds": [run["seed"] for run in runs],
            "seed_count": len(runs),
            "base_bet": runs[0]["base_bet"],
            "deck_count": runs[0]["deck_count"],
            "strategy_implementation": runs[0]["strategy_implementation"],
        }
    )
    rounds = aggregate["completed_rounds"]
    original_hands = aggregate["total_original_hands"]
    aggregate.update(
        {
            "player_net_per_round": _rate(aggregate["net_player_result"], rounds),
            "player_net_per_initial_wager": _rate(
                aggregate["net_player_result"], aggregate["initial_wagered"]
            ),
            "player_net_per_total_wager": _rate(
                aggregate["net_player_result"], aggregate["total_wagered"]
            ),
            "player_blackjack_rate": _rate(
                aggregate["player_blackjacks"], original_hands
            ),
            "double_actions_per_round": _rate(aggregate["double_actions"], rounds),
            "split_actions_per_round": _rate(aggregate["split_actions"], rounds),
            "initial_pair_opportunity_rate": _rate(
                aggregate["initial_pair_opportunities"], rounds
            ),
            "split_rate_given_initial_pair": _rate(
                aggregate["split_actions"], aggregate["initial_pair_opportunities"]
            ),
            "dealer_blackjack_rate": _rate(aggregate["dealer_blackjacks"], rounds),
            "average_cards_drawn_per_round": _rate(
                aggregate["total_cards_drawn"], rounds
            ),
        }
    )
    aggregate["pair_diagnostics"] = _aggregate_pair_diagnostics(runs, rounds)
    aggregate["streak_metrics"] = aggregate_streak_metrics(
        [run["streak_metrics"] for run in runs]
    )
    streaks = aggregate["streak_metrics"]
    aggregate.update(
        {
            "winning_rounds": streaks["winning_rounds"],
            "losing_rounds": streaks["losing_rounds"],
            "push_rounds": streaks["push_rounds"],
            "win_rate": streaks["win_rate"],
            "loss_rate": streaks["loss_rate"],
            "push_rate": streaks["push_rate"],
            "total_win_streaks": streaks["win_streaks"]["streak_count"],
            "mean_win_streak": streaks["win_streaks"]["mean"],
            "median_win_streak": streaks["win_streaks"]["median"],
            "p95_win_streak": streaks["win_streaks"]["p95"],
            "max_win_streak": streaks["win_streaks"]["maximum"],
            "total_loss_streaks": streaks["loss_streaks"]["streak_count"],
            "mean_loss_streak": streaks["loss_streaks"]["mean"],
            "median_loss_streak": streaks["loss_streaks"]["median"],
            "p95_loss_streak": streaks["loss_streaks"]["p95"],
            "max_loss_streak": streaks["loss_streaks"]["maximum"],
        }
    )
    one2six_runs = [run for run in runs if run["one2six_invariant_passed"] is not None]
    aggregate["one2six_invariant_passed"] = (
        all(run["one2six_invariant_passed"] for run in one2six_runs)
        if one2six_runs
        else None
    )
    for field_name in (
        "one2six_accepted_discard_batch_count",
        "one2six_ejection_count",
        "one2six_fallback_count",
    ):
        values = [run[field_name] for run in runs if run[field_name] is not None]
        aggregate[field_name] = sum(values) if values else None
    ejections = aggregate["one2six_ejection_count"]
    fallbacks = aggregate["one2six_fallback_count"]
    aggregate["one2six_fallback_rate"] = (
        _rate(fallbacks, ejections)
        if fallbacks is not None and ejections is not None
        else None
    )
    return aggregate


def _aggregate_pair_diagnostics(
    runs: list[dict[str, Any]],
    rounds: int,
) -> dict[str, Any]:
    recorder = _empty_pair_metrics(rounds)
    for run in runs:
        pair = run["pair_diagnostics"]
        for field_name in (
            "initial_pair_opportunities",
            "actual_split_actions",
            "initial_pairs_not_split",
            "expected_split_missing_legal_action",
            "actual_split_without_initial_pair",
        ):
            recorder[field_name] += pair[field_name]
        for action in NON_SPLIT_ACTIONS:
            recorder["non_split_actions"][action] += pair["non_split_actions"][action]
        for category in PAIR_CATEGORIES:
            source_cell = pair["categories"][category]
            target_cell = recorder["categories"][category]
            _add_pair_cell(target_cell, source_cell)

    opportunities = recorder["initial_pair_opportunities"]
    splits = recorder["actual_split_actions"]
    recorder["initial_pair_opportunity_rate"] = _rate(opportunities, rounds)
    recorder["actual_split_rate_per_round"] = _rate(splits, rounds)
    recorder["split_rate_given_initial_pair"] = _rate(splits, opportunities)
    recorder["categories"] = {
        category: _pair_cell_metrics(cell, rounds)
        for category, cell in recorder["categories"].items()
    }
    validate_pair_reconciliation(recorder)
    return recorder


def aggregate_streak_metrics(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool finalized streaks without joining them across seed boundaries."""
    win_frequency: Counter[int] = Counter()
    loss_frequency: Counter[int] = Counter()
    for metrics in seed_metrics:
        validate_streak_reconciliation(metrics)
        win_frequency.update(metrics["win_streaks"]["frequency"])
        loss_frequency.update(metrics["loss_streaks"]["frequency"])

    rounds = sum(metrics["rounds"] for metrics in seed_metrics)
    wins = sum(metrics["winning_rounds"] for metrics in seed_metrics)
    losses = sum(metrics["losing_rounds"] for metrics in seed_metrics)
    pushes = sum(metrics["push_rounds"] for metrics in seed_metrics)
    summary = {
        "rounds": rounds,
        "winning_rounds": wins,
        "losing_rounds": losses,
        "push_rounds": pushes,
        "win_rate": _rate(wins, rounds),
        "loss_rate": _rate(losses, rounds),
        "push_rate": _rate(pushes, rounds),
        "win_streaks": streak_frequency_summary(win_frequency),
        "loss_streaks": streak_frequency_summary(loss_frequency),
        "mean_per_seed_maximum_win_streak": _mean_seed_maximum(
            seed_metrics,
            "win_streaks",
        ),
        "mean_per_seed_maximum_loss_streak": _mean_seed_maximum(
            seed_metrics,
            "loss_streaks",
        ),
        "seed_runs": len(seed_metrics),
    }
    validate_streak_reconciliation(summary)
    return summary


def _mean_seed_maximum(seed_metrics: list[dict[str, Any]], streak_type: str) -> float:
    if not seed_metrics:
        return 0.0
    maxima = [metrics[streak_type]["maximum"] or 0 for metrics in seed_metrics]
    return sum(maxima) / len(maxima)


def _streak_source_differences(
    aggregates: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    physical = aggregates["physical_iid"]
    one2six = aggregates["one2six"]
    return {
        "win_rate": one2six["win_rate"] - physical["win_rate"],
        "loss_rate": one2six["loss_rate"] - physical["loss_rate"],
        "push_rate": one2six["push_rate"] - physical["push_rate"],
        "mean_win_streak": (
            _value_or_zero(one2six["mean_win_streak"])
            - _value_or_zero(physical["mean_win_streak"])
        ),
        "mean_loss_streak": (
            _value_or_zero(one2six["mean_loss_streak"])
            - _value_or_zero(physical["mean_loss_streak"])
        ),
        "maximum_win_streak": (
            _value_or_zero(one2six["max_win_streak"])
            - _value_or_zero(physical["max_win_streak"])
        ),
        "maximum_loss_streak": (
            _value_or_zero(one2six["max_loss_streak"])
            - _value_or_zero(physical["max_loss_streak"])
        ),
    }


def _value_or_zero(value: int | float | None) -> int | float:
    return 0 if value is None else value


def _seed_level_uncertainty(runs_by_source: SourceRuns) -> dict[str, Any]:
    rate_fields = {
        "player_blackjack_rate": "player_blackjack_rate",
        "double_action_rate": "double_actions_per_round",
        "initial_pair_opportunity_rate": "initial_pair_opportunity_rate",
        "actual_split_action_rate": "split_actions_per_round",
        "split_rate_given_initial_pair": "split_rate_given_initial_pair",
    }
    sources: dict[str, Any] = {}
    for source_name, runs in runs_by_source.items():
        sources[source_name] = {
            "player_edge_per_initial_wager": student_t_summary(
                [run["player_net_per_initial_wager"] for run in runs]
            ),
            "mean_net_player_result": sum(run["net_player_result"] for run in runs)
            / len(runs),
            "event_rates": {
                label: student_t_summary([run[field] for run in runs])
                for label, field in rate_fields.items()
            },
        }

    physical_runs = runs_by_source["physical_iid"]
    one2six_runs = runs_by_source["one2six"]
    paired = {
        "player_edge_per_initial_wager": paired_difference_summary(
            physical_runs,
            one2six_runs,
            "player_net_per_initial_wager",
        ),
        "event_rates": {
            label: paired_difference_summary(physical_runs, one2six_runs, field)
            for label, field in rate_fields.items()
        },
    }
    return {
        "inferential_unit": "independent seed run",
        "confidence_interval": "two-sided 95% Student-t",
        "sources": sources,
        "paired_one2six_minus_physical_iid": paired,
    }


def _comparison_rows() -> tuple[tuple[str, str, str], ...]:
    return (
        ("Rounds", "completed_rounds", "count"),
        ("Net player result", "net_player_result", "money"),
        ("Net casino result", "net_casino_result", "money"),
        ("Net per round", "player_net_per_round", "money"),
        ("Initial wagered", "initial_wagered", "money"),
        ("Additional action wagered", "additional_action_wagered", "money"),
        ("Total wagered", "total_wagered", "money"),
        ("Player blackjack rate", "player_blackjack_rate", "rate"),
        ("Double rate", "double_actions_per_round", "rate"),
        ("Initial pair rate", "initial_pair_opportunity_rate", "rate"),
        ("Actual split rate", "split_actions_per_round", "rate"),
        ("Split given pair", "split_rate_given_initial_pair", "rate"),
        ("Dealer blackjack rate", "dealer_blackjack_rate", "rate"),
        ("Winning rounds", "winning_rounds", "count"),
        ("Losing rounds", "losing_rounds", "count"),
        ("Push rounds", "push_rounds", "count"),
        ("Win rate", "win_rate", "rate"),
        ("Loss rate", "loss_rate", "rate"),
        ("Push rate", "push_rate", "rate"),
        ("Total win streaks", "total_win_streaks", "count"),
        ("Mean win streak", "mean_win_streak", "decimal"),
        ("Median win streak", "median_win_streak", "decimal"),
        ("95th percentile win streak", "p95_win_streak", "decimal"),
        ("Maximum win streak", "max_win_streak", "count"),
        ("Total loss streaks", "total_loss_streaks", "count"),
        ("Mean loss streak", "mean_loss_streak", "decimal"),
        ("Median loss streak", "median_loss_streak", "decimal"),
        ("95th percentile loss streak", "p95_loss_streak", "decimal"),
        ("Maximum loss streak", "max_loss_streak", "count"),
    )


def _write_summary_csv(path: Path, aggregates: dict[str, dict[str, Any]]) -> None:
    physical = aggregates["physical_iid"]
    one2six = aggregates["one2six"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["metric", "physical_iid", "one2six", "one2six_minus_iid", "unit"]
        )
        for label, field_name, kind in _comparison_rows():
            difference = one2six[field_name] - physical[field_name]
            unit = kind
            if kind == "rate":
                difference *= 100.0
                unit = "rate; difference in percentage points"
            writer.writerow(
                [label, physical[field_name], one2six[field_name], difference, unit]
            )


def _write_per_seed_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = (
        "source",
        "seed",
        "completed_rounds",
        "net_player_result",
        "net_casino_result",
        "player_net_per_round",
        "player_net_per_initial_wager",
        "player_net_per_total_wager",
        "player_blackjack_rate",
        "double_actions_per_round",
        "initial_pair_opportunity_rate",
        "split_actions_per_round",
        "split_rate_given_initial_pair",
    )
    statistics_fields = (
        "mean_round_profit",
        "sample_variance_round_profit",
        "sample_standard_deviation_round_profit",
        "naive_round_se",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                *fields,
                *statistics_fields,
                "naive_ci_lower",
                "naive_ci_upper",
            ],
        )
        writer.writeheader()
        for run in runs:
            statistics = run["round_profit_statistics"]
            interval = statistics["naive_normal_95_ci"]
            row = {field_name: run[field_name] for field_name in fields}
            row.update(
                {field_name: statistics[field_name] for field_name in statistics_fields}
            )
            row["naive_ci_lower"] = interval[0] if interval else None
            row["naive_ci_upper"] = interval[1] if interval else None
            writer.writerow(row)


def _write_pair_summary_csv(
    path: Path,
    aggregates: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "source",
                "pair",
                "opportunities",
                "opportunities_per_round",
                "actual_splits",
                "split_rate_given_pair",
                "non_split_hit",
                "non_split_stand",
                "non_split_double",
            ]
        )
        for source_name in SOURCE_NAMES:
            categories = aggregates[source_name]["pair_diagnostics"]["categories"]
            for category in PAIR_CATEGORIES:
                cell = categories[category]
                writer.writerow(
                    [
                        source_name,
                        category,
                        cell["opportunities"],
                        cell["opportunities_per_round"],
                        cell["actual_splits"],
                        cell["split_rate_given_pair"],
                        cell["non_split_actions"]["hit"],
                        cell["non_split_actions"]["stand"],
                        cell["non_split_actions"]["double"],
                    ]
                )


def _write_pair_upcard_csv(
    path: Path,
    aggregates: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "source",
                "pair",
                "dealer_upcard",
                "opportunities",
                "actual_splits",
                "split_rate_given_pair_upcard",
                "non_split_hit",
                "non_split_stand",
                "non_split_double",
            ]
        )
        for source_name in SOURCE_NAMES:
            categories = aggregates[source_name]["pair_diagnostics"]["categories"]
            for category in PAIR_CATEGORIES:
                for upcard in UPCARD_CATEGORIES:
                    cell = categories[category]["by_upcard"][upcard]
                    writer.writerow(
                        [
                            source_name,
                            category,
                            upcard,
                            cell["opportunities"],
                            cell["actual_splits"],
                            cell["split_rate_given_pair_upcard"],
                            cell["non_split_actions"]["hit"],
                            cell["non_split_actions"]["stand"],
                            cell["non_split_actions"]["double"],
                        ]
                    )


def _write_streak_summary_csv(
    path: Path,
    runs: list[dict[str, Any]],
    aggregates: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "source",
        "seed",
        "rounds",
        "winning_rounds",
        "losing_rounds",
        "push_rounds",
        "win_rate",
        "loss_rate",
        "push_rate",
        "win_streak_count",
        "loss_streak_count",
        "mean_win_streak",
        "median_win_streak",
        "p95_win_streak",
        "max_win_streak",
        "mean_loss_streak",
        "median_loss_streak",
        "p95_loss_streak",
        "max_loss_streak",
        "mean_per_seed_max_win_streak",
        "mean_per_seed_max_loss_streak",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                _streak_summary_row(
                    source=run["source"],
                    seed=str(run["seed"]),
                    metrics=run["streak_metrics"],
                )
            )
        for source_name in SOURCE_NAMES:
            writer.writerow(
                _streak_summary_row(
                    source=source_name,
                    seed="aggregate",
                    metrics=aggregates[source_name]["streak_metrics"],
                )
            )


def _streak_summary_row(
    *,
    source: str,
    seed: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    win = metrics["win_streaks"]
    loss = metrics["loss_streaks"]
    return {
        "source": source,
        "seed": seed,
        "rounds": metrics["rounds"],
        "winning_rounds": metrics["winning_rounds"],
        "losing_rounds": metrics["losing_rounds"],
        "push_rounds": metrics["push_rounds"],
        "win_rate": metrics["win_rate"],
        "loss_rate": metrics["loss_rate"],
        "push_rate": metrics["push_rate"],
        "win_streak_count": win["streak_count"],
        "loss_streak_count": loss["streak_count"],
        "mean_win_streak": win["mean"],
        "median_win_streak": win["median"],
        "p95_win_streak": win["p95"],
        "max_win_streak": win["maximum"],
        "mean_loss_streak": loss["mean"],
        "median_loss_streak": loss["median"],
        "p95_loss_streak": loss["p95"],
        "max_loss_streak": loss["maximum"],
        "mean_per_seed_max_win_streak": metrics.get("mean_per_seed_maximum_win_streak"),
        "mean_per_seed_max_loss_streak": metrics.get(
            "mean_per_seed_maximum_loss_streak"
        ),
    }


def _write_streak_distribution_csv(
    path: Path,
    runs: list[dict[str, Any]],
    aggregates: dict[str, dict[str, Any]],
) -> None:
    entries = [(run["source"], str(run["seed"]), run["streak_metrics"]) for run in runs]
    entries.extend(
        (
            source_name,
            "aggregate",
            aggregates[source_name]["streak_metrics"],
        )
        for source_name in SOURCE_NAMES
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "source",
                "seed_or_aggregate",
                "streak_type",
                "streak_length",
                "frequency",
                "proportion_of_same_type_streaks",
            ]
        )
        for source_name, seed, metrics in entries:
            for streak_type in ("win", "loss"):
                summary = metrics[f"{streak_type}_streaks"]
                streak_count = summary["streak_count"]
                for length, frequency in summary["frequency"].items():
                    writer.writerow(
                        [
                            source_name,
                            seed,
                            streak_type,
                            length,
                            frequency,
                            _rate(frequency, streak_count),
                        ]
                    )


def _summary_markdown(
    config: SingleBoxGameValidationConfig,
    aggregates: dict[str, dict[str, Any]],
    uncertainty: dict[str, Any],
) -> str:
    physical = aggregates["physical_iid"]
    one2six = aggregates["one2six"]
    source_uncertainty = uncertainty["sources"]
    paired = uncertainty["paired_one2six_minus_physical_iid"]
    physical_edge = source_uncertainty["physical_iid"]["player_edge_per_initial_wager"]
    one2six_edge = source_uncertainty["one2six"]["player_edge_per_initial_wager"]
    paired_edge = paired["player_edge_per_initial_wager"]
    lines = [
        "# Single-Box Game Validation",
        "",
        VALIDATION_NOTE,
        "",
        (
            f"Six decks, one box, flat ${config.base_bet:g} base wager, "
            f"{config.rounds:,} rounds per seed, {len(config.seeds)} independent seeds."
        ),
        "",
        "| Metric | Physical IID | One2Six | One2Six minus IID |",
        "|---|---:|---:|---:|",
        _markdown_row(
            "Rounds",
            physical["completed_rounds"],
            one2six["completed_rounds"],
            one2six["completed_rounds"] - physical["completed_rounds"],
            "count",
        ),
        _markdown_row(
            "Mean player edge",
            physical_edge["mean"],
            one2six_edge["mean"],
            paired_edge["mean"],
            "rate",
        ),
        (
            "| 95% seed-level CI | "
            f"{_format_interval(physical_edge['student_t_95_ci'])} | "
            f"{_format_interval(one2six_edge['student_t_95_ci'])} | "
            f"{_format_interval(paired_edge['student_t_95_ci'])} |"
        ),
        _markdown_row(
            "Net player result",
            physical["net_player_result"],
            one2six["net_player_result"],
            one2six["net_player_result"] - physical["net_player_result"],
            "money",
        ),
    ]
    for label, field_name in (
        ("Player blackjack rate", "player_blackjack_rate"),
        ("Double rate", "double_actions_per_round"),
        ("Initial pair rate", "initial_pair_opportunity_rate"),
        ("Actual split rate", "split_actions_per_round"),
        ("Split given pair", "split_rate_given_initial_pair"),
    ):
        lines.append(
            _markdown_row(
                label,
                physical[field_name],
                one2six[field_name],
                one2six[field_name] - physical[field_name],
                "rate",
            )
        )

    difference_rates = paired["event_rates"]
    split_decomposition = split_count_decomposition(physical, one2six)
    opportunity_component = split_decomposition[
        "opportunity_volume_component_at_iid_conditional_rate"
    ]
    conditional_component = split_decomposition[
        "conditional_mix_component_at_one2six_pair_count"
    ]
    lines.extend(
        [
            "",
            "## Paired Source Differences",
            "",
            "| Metric | Mean One2Six minus IID | 95% seed-level CI |",
            "|---|---:|---:|",
            _difference_markdown_row(
                "Player edge",
                paired["player_edge_per_initial_wager"],
            ),
            _difference_markdown_row(
                "Initial pair rate",
                difference_rates["initial_pair_opportunity_rate"],
            ),
            _difference_markdown_row(
                "Actual split rate",
                difference_rates["actual_split_action_rate"],
            ),
            "",
            (
                "Split-count difference: "
                f"{split_decomposition['actual_split_count_difference']:+,}. "
                "Opportunity-volume component: "
                f"{opportunity_component:+,.2f}; "
                "conditional-mix component: "
                f"{conditional_component:+,.2f}."
            ),
            "",
            "## Pair Categories",
            "",
            "| Pair | IID opportunities | One2Six opportunities | IID splits | "
            "One2Six splits | IID split given pair | One2Six split given pair |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for category in PAIR_CATEGORIES:
        physical_cell = physical["pair_diagnostics"]["categories"][category]
        one2six_cell = one2six["pair_diagnostics"]["categories"][category]
        lines.append(
            f"| {category} | {physical_cell['opportunities']:,} | "
            f"{one2six_cell['opportunities']:,} | "
            f"{physical_cell['actual_splits']:,} | "
            f"{one2six_cell['actual_splits']:,} | "
            f"{physical_cell['split_rate_given_pair']:.4%} | "
            f"{one2six_cell['split_rate_given_pair']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Monetary Round Streaks",
            "",
            "| Metric | Physical IID | One2Six | One2Six minus IID |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, field_name, kind in (
        ("Winning rounds", "winning_rounds", "count"),
        ("Losing rounds", "losing_rounds", "count"),
        ("Push rounds", "push_rounds", "count"),
        ("Win rate", "win_rate", "rate"),
        ("Loss rate", "loss_rate", "rate"),
        ("Push rate", "push_rate", "rate"),
        ("Total win streaks", "total_win_streaks", "count"),
        ("Mean win streak", "mean_win_streak", "decimal"),
        ("Median win streak", "median_win_streak", "decimal"),
        ("95th percentile win streak", "p95_win_streak", "decimal"),
        ("Maximum win streak", "max_win_streak", "count"),
        ("Total loss streaks", "total_loss_streaks", "count"),
        ("Mean loss streak", "mean_loss_streak", "decimal"),
        ("Median loss streak", "median_loss_streak", "decimal"),
        ("95th percentile loss streak", "p95_loss_streak", "decimal"),
        ("Maximum loss streak", "max_loss_streak", "count"),
    ):
        lines.append(
            _markdown_row(
                label,
                physical[field_name],
                one2six[field_name],
                one2six[field_name] - physical[field_name],
                kind,
            )
        )
    lines.extend(
        [
            "",
            "Physical IID theoretical initial-pair rate: "
            f"{PHYSICAL_IID_PAIR_OPPORTUNITY_RATE:.6%} (`25 / 169`).",
            "",
            "Round-level normal intervals are descriptive only. Independent seed runs "
            "are the primary uncertainty unit because One2Six rounds may be serially "
            "dependent.",
            "",
            "![Signed monetary streak histogram](signed_streak_length_histogram.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _empty_pair_cell(*, include_upcards: bool) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "opportunities": 0,
        "actual_splits": 0,
        "non_split_actions": {action: 0 for action in NON_SPLIT_ACTIONS},
    }
    if include_upcards:
        cell["by_upcard"] = {
            upcard: _empty_pair_cell(include_upcards=False)
            for upcard in UPCARD_CATEGORIES
        }
    return cell


def _empty_pair_metrics(rounds: int) -> dict[str, Any]:
    return {
        "total_original_rounds": rounds,
        "initial_pair_opportunities": 0,
        "initial_pair_opportunity_rate": 0.0,
        "actual_split_actions": 0,
        "actual_split_rate_per_round": 0.0,
        "split_rate_given_initial_pair": 0.0,
        "initial_pairs_not_split": 0,
        "non_split_actions": {action: 0 for action in NON_SPLIT_ACTIONS},
        "expected_split_missing_legal_action": 0,
        "actual_split_without_initial_pair": 0,
        "categories": {
            category: _empty_pair_cell(include_upcards=True)
            for category in PAIR_CATEGORIES
        },
    }


def _pair_cell_metrics(cell: dict[str, Any], rounds: int) -> dict[str, Any]:
    result = {
        "opportunities": cell["opportunities"],
        "opportunities_per_round": _rate(cell["opportunities"], rounds),
        "actual_splits": cell["actual_splits"],
        "split_rate_given_pair": _rate(cell["actual_splits"], cell["opportunities"]),
        "non_split_actions": dict(cell["non_split_actions"]),
    }
    if "by_upcard" in cell:
        result["by_upcard"] = {
            upcard: {
                "opportunities": upcard_cell["opportunities"],
                "actual_splits": upcard_cell["actual_splits"],
                "split_rate_given_pair_upcard": _rate(
                    upcard_cell["actual_splits"], upcard_cell["opportunities"]
                ),
                "non_split_actions": dict(upcard_cell["non_split_actions"]),
            }
            for upcard, upcard_cell in cell["by_upcard"].items()
        }
    return result


def _add_pair_cell(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["opportunities"] += source["opportunities"]
    target["actual_splits"] += source["actual_splits"]
    for action in NON_SPLIT_ACTIONS:
        target["non_split_actions"][action] += source["non_split_actions"][action]
    for upcard in UPCARD_CATEGORIES:
        target_upcard = target["by_upcard"][upcard]
        source_upcard = source["by_upcard"][upcard]
        target_upcard["opportunities"] += source_upcard["opportunities"]
        target_upcard["actual_splits"] += source_upcard["actual_splits"]
        for action in NON_SPLIT_ACTIONS:
            target_upcard["non_split_actions"][action] += source_upcard[
                "non_split_actions"
            ][action]


def _published_pair_expects_split(category: str, upcard: str) -> bool:
    pair_value: int | str
    if category == "A-A":
        pair_value = "A"
    elif category == "ten-value pair":
        pair_value = 10
    else:
        pair_value = int(category.split("-", maxsplit=1)[0])
    return PAIR_TOTALS[pair_value][DEALER_UPCARDS.index(upcard)] == "P"


def _metric_definitions() -> dict[str, str]:
    return {
        "initial_pair_opportunities": (
            "Original unsplit two-card hands with equal blackjack point value."
        ),
        "player_blackjacks": (
            "Original unsplit two-card naturals, including dealer-blackjack pushes."
        ),
        "double_actions": "Hands on which the fixed strategy actually doubled.",
        "split_actions": "Actual split actions; no resplitting is enabled.",
        "dealer_blackjacks": (
            "Observed dealer two-card blackjacks; no irrelevant dealer draw is made."
        ),
        "naive_round_se": (
            "Descriptive round-level standard error that does not adjust for serial "
            "dependence."
        ),
        "seed_level_uncertainty": (
            "Student-t interval using independent seed runs as the inferential unit."
        ),
        "monetary_round_streaks": (
            "Classified from total box net result per round; pushes neither count "
            "nor break an open win or loss streak, and seed boundaries finalize it."
        ),
    }


def _markdown_row(
    label: str,
    physical: int | float,
    one2six: int | float,
    difference: int | float,
    kind: str,
) -> str:
    return (
        f"| {label} | {_format_value(physical, kind)} | "
        f"{_format_value(one2six, kind)} | "
        f"{_format_difference(difference, kind)} |"
    )


def _difference_markdown_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {_format_difference(summary['mean'], 'rate')} | "
        f"{_format_interval(summary['student_t_95_ci'])} |"
    )


def _format_value(value: int | float, kind: str) -> str:
    if kind == "count":
        return f"{int(value):,}"
    if kind == "money":
        return f"${value:,.4f}"
    if kind == "decimal":
        return f"{value:,.4f}"
    return f"{100.0 * value:.4f}%"


def _format_difference(value: int | float, kind: str) -> str:
    if kind == "count":
        return f"{int(value):+,}"
    if kind == "money":
        return f"${value:+,.4f}"
    if kind == "decimal":
        return f"{value:+,.4f}"
    return f"{100.0 * value:+.4f} pp"


def _format_interval(interval: list[float] | None) -> str:
    if interval is None:
        return "n/a"
    return f"[{100.0 * interval[0]:.4f}%, {100.0 * interval[1]:.4f}%]"


def _rate(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
