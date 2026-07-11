# SPDX-License-Identifier: GPL-3.0-or-later

"""Extreme frozen-score profitability and insurance feasibility audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Final, Literal

from experiments.conditional_profitability import (
    BandAccumulator,
    _round_observation,
    linear_quantile,
)
from experiments.fading_exclusion_validation import (
    FROZEN_WEIGHTS,
    CohortCounts,
    LedgerCardSource,
    ObservableCohortLedger,
    calculate_fading_state,
)
from experiments.multi_box_counterfactual import _make_source
from experiments.plots import write_extreme_tail_plots
from experiments.single_box_game_validation import student_t_summary
from shufflemaster_sim.card_sources import One2SixCardSource
from shufflemaster_sim.cards import Card
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.hand_values import is_natural_blackjack, split_value
from shufflemaster_sim.state import TableState
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)

Source = Literal["physical_iid", "one2six"]
SOURCE_NAMES: Final[tuple[Source, ...]] = ("physical_iid", "one2six")
DEFAULT_DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (82, 83, 84, 85, 86)
DEFAULT_VALIDATION_SEEDS: Final[tuple[int, ...]] = tuple(range(87, 97))
QUANTILES: Final[tuple[float, ...]] = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
)
HIGH_TAILS: Final[tuple[str, ...]] = (
    "high_rich_20",
    "high_rich_10",
    "high_rich_05",
    "high_rich_025",
    "high_rich_01",
)
LOW_TAILS: Final[tuple[str, ...]] = (
    "low_rich_20",
    "low_rich_10",
    "low_rich_05",
    "low_rich_025",
    "low_rich_01",
)
TAIL_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(
    zip(HIGH_TAILS, LOW_TAILS, strict=True)
)
SLICE_NAMES: Final[tuple[str, ...]] = (
    "lowest_0_1",
    "1_2_5",
    "2_5_5",
    "5_10",
    "10_20",
    "20_40",
    "40_60",
    "60_80",
    "80_90",
    "90_95",
    "95_97_5",
    "97_5_99",
    "99_100",
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
    "hidden_telemetry",
)


@dataclass(frozen=True, slots=True)
class ExtremeTailProfitabilityConfig:
    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS
    validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS
    development_rounds_per_seed: int = 20_000
    validation_rounds_per_seed: int = 100_000
    burn_in_rounds: int = 1_000
    deck_count: int = 6
    base_bet: float = 10.0
    high_rich_quantiles: tuple[float, ...] = (0.20, 0.10, 0.05, 0.025, 0.01)
    neutral_quantiles: tuple[float, float] = (0.40, 0.60)
    current_rack_weight: float = 1.00
    returned_1_15_weight: float = 0.75
    returned_16_50_weight: float = 0.40
    returned_51_100_weight: float = 0.20
    returned_over_100_weight: float = 0.00
    output_dir: Path = Path("experiments/outputs/extreme_tail_profitability")

    def __post_init__(self) -> None:
        if not self.development_seeds or not self.validation_seeds:
            raise ValueError("development and validation seeds must be nonempty.")
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("development seeds must be unique.")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation seeds must be unique.")
        if set(self.development_seeds) & set(self.validation_seeds):
            raise ValueError("development and validation seeds must be disjoint.")
        if (
            self.development_rounds_per_seed <= 0
            or self.validation_rounds_per_seed <= 0
        ):
            raise ValueError("round counts must be positive.")
        if self.burn_in_rounds < 0 or self.base_bet <= 0:
            raise ValueError("burn-in must be nonnegative and base bet positive.")
        if self.deck_count != 6:
            raise ValueError("extreme-tail audit requires exactly six decks.")
        if self.high_rich_quantiles != tuple(
            sorted(self.high_rich_quantiles, reverse=True)
        ):
            raise ValueError("high-rich quantiles must be descending.")
        if any(not 0 < value < 1 for value in self.high_rich_quantiles):
            raise ValueError("quantiles must lie strictly inside zero and one.")
        if not 0 < self.neutral_quantiles[0] < self.neutral_quantiles[1] < 1:
            raise ValueError("neutral quantiles must be ordered inside zero and one.")
        if self.weights != FROZEN_WEIGHTS:
            raise ValueError("weights must match the frozen documented kernel.")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "current_rack": self.current_rack_weight,
            "returned_1_15": self.returned_1_15_weight,
            "returned_16_50": self.returned_16_50_weight,
            "returned_51_100": self.returned_51_100_weight,
            "returned_over_100": self.returned_over_100_weight,
        }


@dataclass(frozen=True, slots=True)
class TailCutpoints:
    values: Mapping[str, float]

    def __getitem__(self, name: str) -> float:
        return self.values[name]


@dataclass(slots=True)
class InsuranceAccumulator:
    opportunities: int = 0
    events: int = 0
    decision_shift_sum: float = 0.0

    def add(self, event: bool, decision_shift: float) -> None:
        self.opportunities += 1
        self.events += event
        self.decision_shift_sum += decision_shift

    def metrics(self, *, payout_multiplier: float, threshold: float) -> dict[str, Any]:
        probability = self.events / self.opportunities if self.opportunities else None
        return {
            "opportunities": self.opportunities,
            "insured_events": self.events,
            "event_probability": probability,
            "break_even_probability": threshold,
            "probability_minus_break_even": probability - threshold
            if probability is not None
            else None,
            "implied_ev_per_unit": payout_multiplier * probability - 1
            if probability is not None
            else None,
            "mean_decision_shift": self.decision_shift_sum / self.opportunities
            if self.opportunities
            else None,
        }


@dataclass(frozen=True, slots=True)
class InsuranceSnapshot:
    game: CasinoBlackjackGame
    table: TableState
    source: LedgerCardSource


def freeze_cutpoints(scores: Sequence[float]) -> TailCutpoints:
    """Freeze all cutpoints from score values only."""
    if not scores:
        raise ValueError("development scores must be nonempty.")
    return TailCutpoints(
        {quantile_name(value): linear_quantile(scores, value) for value in QUANTILES}
    )


def quantile_name(value: float) -> str:
    return f"q{str(value * 100).replace('.', '_').replace('_0', '')}"


def nested_sets(score: float, cutpoints: TailCutpoints) -> tuple[str, ...]:
    labels = []
    for label, quantile in zip(HIGH_TAILS, (20, 10, 5, 2.5, 1), strict=True):
        if score <= cutpoints[quantile_name(quantile / 100)]:
            labels.append(label)
    if cutpoints["q40"] <= score <= cutpoints["q60"]:
        labels.append("neutral")
    for label, quantile in zip(LOW_TAILS, (80, 90, 95, 97.5, 99), strict=True):
        if score >= cutpoints[quantile_name(quantile / 100)]:
            labels.append(label)
    return tuple(labels)


def disjoint_slice(score: float, cutpoints: TailCutpoints) -> str:
    boundaries = (
        cutpoints["q1"],
        cutpoints["q2_5"],
        cutpoints["q5"],
        cutpoints["q10"],
        cutpoints["q20"],
        cutpoints["q40"],
        cutpoints["q60"],
        cutpoints["q80"],
        cutpoints["q90"],
        cutpoints["q95"],
        cutpoints["q97_5"],
        cutpoints["q99"],
    )
    for index, boundary in enumerate(boundaries):
        if score <= boundary:
            return SLICE_NAMES[index]
    return SLICE_NAMES[-1]


def insurance_ev(probability: float, payout_total_multiplier: float) -> float:
    return payout_total_multiplier * probability - 1.0


def decline_even_money_value(probability: float) -> float:
    return 1.5 * (1.0 - probability)


def positive_ev_gate(
    one: Mapping[str, Any],
    iid: Mapping[str, Any],
    difference: Mapping[str, Any],
    contrast: Mapping[str, Any],
    *,
    minimum_frequency: float = 0.005,
) -> bool:
    return bool(
        one["mean"] > 0
        and _ci_positive(one["student_t_95_ci"])
        and one["positive_seeds"] >= 8
        and one["frequency"] >= minimum_frequency
        and not (
            iid["mean"] > 0
            and _ci_positive(iid["student_t_95_ci"])
            and iid["positive_seeds"] >= 8
        )
        and difference["mean"] > 0
        and _ci_positive(difference["student_t_95_ci"])
        and contrast["mean"] > 0
        and _ci_positive(contrast["student_t_95_ci"])
    )


def insurance_gate(
    row: Mapping[str, Any],
    iid: Mapping[str, Any],
    *,
    threshold: float,
    minimum_opportunities: int = 100,
) -> bool:
    return bool(
        row["mean_probability"] > threshold
        and isinstance(row["student_t_95_ci"], list)
        and row["student_t_95_ci"][0] > threshold
        and row["seeds_above_threshold"] >= 8
        and row["minimum_seed_opportunities"] >= minimum_opportunities
        and not (
            iid["mean_probability"] > threshold
            and iid["student_t_95_ci"][0] > threshold
        )
    )


def planning_precision(
    seed_edges: Sequence[float], current_rounds: int
) -> dict[str, Any]:
    summary = student_t_summary(list(seed_edges))
    sd = summary["sample_standard_deviation"]
    half_width = None
    mde = None
    if sd is not None:
        half_width = summary["student_t_critical"] * sd / sqrt(len(seed_edges))
        mde = (summary["student_t_critical"] + 0.841621234) * sd / sqrt(len(seed_edges))
    return {
        "observed_seed_standard_deviation": sd,
        "confidence_interval_half_width": half_width,
        "approximate_mde_80_percent_power": mde,
        **{
            f"approximate_rounds_for_{int(target * 10000)}bp": (
                int(current_rounds * (mde / target) ** 2 + 0.999999)
                if mde is not None
                else None
            )
            for target in (0.0025, 0.005, 0.01)
        },
        "planning_note": (
            "Approximation uses independent-seed variance; not round-IID inference."
        ),
    }


def run_extreme_tail_profitability(
    config: ExtremeTailProfitabilityConfig,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    development_scores = []
    for seed in config.development_seeds:
        development_scores.extend(_collect_scores(config, seed))
    cutpoints = freeze_cutpoints(development_scores)
    per_seed_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    insurance_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for source in SOURCE_NAMES:
        for seed in config.validation_seeds:
            result = _run_validation_seed(config, source, seed, cutpoints)
            per_seed_rows.extend(result["tails"])
            slice_rows.extend(result["slices"])
            insurance_rows.extend(result["insurance"])
            opportunity_rows.extend(result["opportunities"])
            run_rows.extend(result["runs"])
    aggregate_rows = _aggregate_tail_rows(per_seed_rows, config)
    aggregate_slices = _aggregate_slice_rows(slice_rows)
    shape_diagnostics = _shape_diagnostics(slice_rows, config)
    contrasts, paired = _contrasts(per_seed_rows, config)
    gates = _positive_gates(aggregate_rows, contrasts, paired)
    insurance = _aggregate_insurance(insurance_rows, config)
    precision = _precision_rows(per_seed_rows, aggregate_rows, config)
    plot_paths = write_extreme_tail_plots(
        config.output_dir,
        aggregate_rows,
        aggregate_slices,
        paired,
        opportunity_rows,
        insurance,
    )
    summary = {
        "experiment": "extreme_tail_profitability_and_insurance_feasibility",
        "config": _config_payload(config),
        "cutpoints": {
            "values": dict(cutpoints.values),
            "source": "one2six_development",
            "action_outcomes_used": False,
        },
        "nested_tail_profitability": aggregate_rows,
        "disjoint_slice_profitability": aggregate_slices,
        "disjoint_slice_shape_diagnostics": shape_diagnostics,
        "tail_contrasts": contrasts,
        "paired_source_tail_differences": paired,
        "positive_ev_state_gate": gates,
        "insurance_feasibility": insurance,
        "opportunity_frequency": opportunity_rows,
        "precision": precision,
        "plot_paths": plot_paths,
        "hidden_state_exported": False,
    }
    _validate_privacy(summary)
    _write_outputs(
        config,
        summary,
        cutpoints,
        aggregate_rows,
        per_seed_rows,
        aggregate_slices,
        contrasts,
        paired,
        gates,
        opportunity_rows,
        run_rows,
        precision,
        insurance,
    )
    return summary


def _new_trajectory(
    config: ExtremeTailProfitabilityConfig, source_name: Source, seed: int
):
    source = LedgerCardSource(
        _make_source(source_name, config.deck_count, seed), ObservableCohortLedger()
    )
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            base_bet=config.base_bet,
            box_count=1,
            box_bets={1: config.base_bet},
            deck_count=6,
        )
    )
    return source, game, PublishedApproxCasinoStrategy()


def _prebet_state(config, source, game):
    return calculate_fading_state(
        current_rack=CohortCounts.from_cards(game.pending_discard_rack),
        returned_by_band=source.ledger.active_by_band(source.draw_count),
        weights=config.weights,
    )


def _collect_scores(config: ExtremeTailProfitabilityConfig, seed: int) -> list[float]:
    source, game, strategy = _new_trajectory(config, "one2six", seed)
    for round_index in range(config.burn_in_rounds):
        game.play_round(round_index=round_index, card_source=source, strategy=strategy)
    scores = []
    for offset in range(config.development_rounds_per_seed):
        scores.append(_prebet_state(config, source, game).predicted_hi_lo_shift)
        game.play_round(
            round_index=config.burn_in_rounds + offset,
            card_source=source,
            strategy=strategy,
        )
    return scores


def _decision_state(config, table, source):
    cards = [card for box in table.boxes for hand in box.hands for card in hand.cards]
    cards.append(table.dealer.cards[0])
    return calculate_fading_state(
        current_rack=CohortCounts.from_cards(cards),
        returned_by_band=source.ledger.active_by_band(source.draw_count),
        weights=config.weights,
    )


def _complete_round(game, table, source, strategy):
    game.settle_immediate_blackjacks(table)
    game.play_player_hands(table, source, strategy)
    game.play_dealer(table, source)
    game.settle(table)
    game.collect_remaining_layout_cards(table)
    game.stage_discard_rack_for_next_round(table)


def _complete_round_with_insurance_event(
    game, table, source, strategy, *, insured_upcard: object | None
) -> Card | None:
    game.settle_immediate_blackjacks(table)
    game.play_player_hands(table, source, strategy)
    game.play_dealer(table, source)
    insured_card = None
    if insured_upcard is not None:
        if len(table.dealer.cards) > 1:
            insured_card = table.dealer.cards[1]
        else:
            isolated_source = deepcopy(source)
            insured_card = isolated_source.draw_card()
    game.settle(table)
    game.collect_remaining_layout_cards(table)
    game.stage_discard_rack_for_next_round(table)
    return insured_card


def insurance_second_card(snapshot: InsuranceSnapshot, strategy) -> Card:
    branch = deepcopy(snapshot)
    branch.game.settle_immediate_blackjacks(branch.table)
    branch.game.play_player_hands(branch.table, branch.source, strategy)
    branch.game.play_dealer(branch.table, branch.source)
    if len(branch.table.dealer.cards) == 1:
        branch.table.dealer.cards.append(branch.source.draw_card())
    return branch.table.dealer.cards[1]


def _run_validation_seed(config, source_name, seed, cutpoints):
    source, game, strategy = _new_trajectory(config, source_name, seed)
    for round_index in range(config.burn_in_rounds):
        game.play_round(round_index=round_index, card_source=source, strategy=strategy)
    tails = {label: BandAccumulator() for label in (*HIGH_TAILS, "neutral", *LOW_TAILS)}
    slices = {label: BandAccumulator() for label in SLICE_NAMES}
    slice_score_sums: dict[str, float] = defaultdict(float)
    insurance: dict[tuple[str, str], InsuranceAccumulator] = defaultdict(
        InsuranceAccumulator
    )
    eligible_indices = {label: [] for label in HIGH_TAILS}
    for offset in range(config.validation_rounds_per_seed):
        score = _prebet_state(config, source, game).predicted_hi_lo_shift
        labels = nested_sets(score, cutpoints)
        slice_name = disjoint_slice(score, cutpoints)
        for label in HIGH_TAILS:
            if label in labels:
                eligible_indices[label].append(offset)
        draw_before = source.draw_count
        source.before_round()
        game.burn_initial_card(source)
        table = game.create_table(config.burn_in_rounds + offset)
        game.deal_initial_cards(table, source)
        initial = (
            table.boxes[0].hands[0].cards[0],
            table.dealer.cards[0],
            table.boxes[0].hands[0].cards[1],
        )
        upcard = split_value(table.dealer.cards[0])
        decision = (
            _decision_state(config, table, source) if upcard in {"A", 10} else None
        )
        second = _complete_round_with_insurance_event(
            game,
            table,
            source,
            strategy,
            insured_upcard=upcard if decision is not None else None,
        )
        if decision is not None and second is not None:
            event_type = "ace_up" if upcard == "A" else "ten_up"
            event = (
                split_value(second) == 10
                if upcard == "A"
                else split_value(second) == "A"
            )
            state_labels = ("all", *labels)
            for label in state_labels:
                shift = (
                    decision.predicted_ten_value_shift
                    if upcard == "A"
                    else decision.predicted_ace_shift
                )
                insurance[(event_type, label)].add(event, shift)
                if upcard == "A" and is_natural_blackjack(
                    table.boxes[0].hands[0].cards
                ):
                    insurance[("even_money", label)].add(event, shift)
        observation = _round_observation(
            score=score,
            band="neutral",
            table=table,
            initial_cards=initial,
            cards_consumed=source.draw_count - draw_before,
        )
        for label in labels:
            tails[label].add(observation)
        slices[slice_name].add(observation)
        slice_score_sums[slice_name] += score
    if isinstance(source.source, One2SixCardSource):
        source.source.assert_invariants(game.pending_discard_rack)
    tail_rows = [
        {
            "source": source_name,
            "seed": seed,
            "state": label,
            "frequency": acc.rounds / config.validation_rounds_per_seed,
            **acc.as_metrics(),
        }
        for label, acc in tails.items()
    ]
    slice_output = [
        {
            "source": source_name,
            "seed": seed,
            "slice": label,
            "slice_rank": index,
            "frequency": acc.rounds / config.validation_rounds_per_seed,
            "mean_score": slice_score_sums[label] / acc.rounds if acc.rounds else None,
            **acc.as_metrics(),
        }
        for index, (label, acc) in enumerate(slices.items())
    ]
    insurance_output = []
    for (kind, label), acc in insurance.items():
        threshold, multiplier = (
            (1 / 3, 3.0) if kind in {"ace_up", "even_money"} else (1 / 11, 11.0)
        )
        insurance_output.append(
            {
                "source": source_name,
                "seed": seed,
                "insurance_type": kind,
                "state": label,
                **acc.metrics(payout_multiplier=multiplier, threshold=threshold),
            }
        )
    opportunities, runs = _opportunity_metrics(
        source_name, seed, eligible_indices, config.validation_rounds_per_seed
    )
    return {
        "tails": tail_rows,
        "slices": slice_output,
        "insurance": insurance_output,
        "opportunities": opportunities,
        "runs": runs,
    }


def _opportunity_metrics(source, seed, indices_by_tail, rounds):
    summaries, rows = [], []
    for tail, indices in indices_by_tail.items():
        waits = [
            right - left for left, right in zip(indices, indices[1:], strict=False)
        ]
        durations = []
        for index in indices:
            if not durations or index != durations[-1][-1] + 1:
                durations.append([index])
            else:
                durations[-1].append(index)
        lengths = [len(run) for run in durations]
        summaries.append(
            {
                "source": source,
                "seed": seed,
                "tail": tail,
                "eligible_rounds": len(indices),
                "eligible_frequency": len(indices) / rounds,
                "mean_wait": _mean(waits),
                "median_wait": _quantile(waits, 0.5),
                "p90_wait": _quantile(waits, 0.9),
                "eligible_runs": len(lengths),
                "mean_duration": _mean(lengths),
                "maximum_duration": max(lengths, default=0),
            }
        )
        rows.extend(
            {"source": source, "seed": seed, "tail": tail, "run_length": length}
            for length in lengths
        )
    return summaries, rows


def _aggregate_tail_rows(rows, config):
    output = []
    for source in SOURCE_NAMES:
        for state in (*HIGH_TAILS, "neutral", *LOW_TAILS):
            matching = [
                r for r in rows if r["source"] == source and r["state"] == state
            ]
            edges = [float(r["player_edge_per_initial_wager"]) for r in matching]
            summary = student_t_summary(edges)
            output.append(
                {
                    "source": source,
                    "state": state,
                    "frequency": sum(float(r["frequency"]) for r in matching)
                    / len(matching),
                    "rounds": sum(int(r["rounds"]) for r in matching),
                    "mean_edge": summary["mean"],
                    "seed_edge_standard_deviation": summary[
                        "sample_standard_deviation"
                    ],
                    "student_t_95_ci": summary["student_t_95_ci"],
                    "minimum": summary["minimum"],
                    "maximum": summary["maximum"],
                    "positive_seeds": sum(v > 0 for v in edges),
                    "negative_seeds": sum(v < 0 for v in edges),
                    "contributing_seeds": len(edges),
                    "mean_round_standard_deviation": _mean(
                        [
                            float(r["round_net_sample_standard_deviation"])
                            / config.base_bet
                            for r in matching
                        ]
                    ),
                }
            )
    return output


def _aggregate_slice_rows(rows):
    output = []
    for source in SOURCE_NAMES:
        for rank, name in enumerate(SLICE_NAMES):
            matching = [r for r in rows if r["source"] == source and r["slice"] == name]
            edges = [float(r["player_edge_per_initial_wager"]) for r in matching]
            summary = student_t_summary(edges)
            output.append(
                {
                    "source": source,
                    "slice": name,
                    "slice_rank": rank,
                    "frequency": _mean([r["frequency"] for r in matching]),
                    "mean_score": _mean([r["mean_score"] for r in matching]),
                    "player_edge": summary["mean"],
                    "student_t_95_ci": summary["student_t_95_ci"],
                    "blackjack_rate": _mean(
                        [r["player_blackjack_rate"] for r in matching]
                    ),
                    "ten_value_rate": _mean(
                        [r["player_card_ten_value_rate"] for r in matching]
                    ),
                    "ace_rate": _mean([r["player_card_ace_rate"] for r in matching]),
                    "low_card_rate": _mean(
                        [r["player_card_low_rate"] for r in matching]
                    ),
                }
            )
    return output


def _contrasts(rows, config):
    index = {(r["source"], r["seed"], r["state"]): r for r in rows}
    contrasts = []
    paired = []
    for high, low in TAIL_PAIRS:
        direct = [
            index[("one2six", seed, high)]["player_edge_per_initial_wager"]
            - index[("physical_iid", seed, high)]["player_edge_per_initial_wager"]
            for seed in config.validation_seeds
        ]
        paired.append(
            {"tail": high, "comparison": "direct_tail_edge", **_summary_signs(direct)}
        )
        for comparison in ("neutral", low):
            source_values = {}
            for source in SOURCE_NAMES:
                values = [
                    index[(source, seed, high)]["player_edge_per_initial_wager"]
                    - index[(source, seed, comparison)]["player_edge_per_initial_wager"]
                    for seed in config.validation_seeds
                ]
                source_values[source] = values
                contrasts.append(
                    {
                        "source": source,
                        "tail": high,
                        "comparison": comparison,
                        **_summary_signs(values),
                    }
                )
            differences = [
                a - b
                for a, b in zip(
                    source_values["one2six"], source_values["physical_iid"], strict=True
                )
            ]
            paired.append(
                {"tail": high, "comparison": comparison, **_summary_signs(differences)}
            )
    return contrasts, paired


def _positive_gates(aggregate, contrasts, paired):
    idx = {(r["source"], r["state"]): r for r in aggregate}
    cidx = {(r["source"], r["tail"], r["comparison"]): r for r in contrasts}
    pidx = {(r["tail"], r["comparison"]): r for r in paired}
    rows = []
    for tail in HIGH_TAILS:
        one = idx[("one2six", tail)]
        iid = idx[("physical_iid", tail)]
        diff = pidx[(tail, "direct_tail_edge")]
        normalized_one = {
            "mean": one["mean_edge"],
            "student_t_95_ci": one["student_t_95_ci"],
            "positive_seeds": one["positive_seeds"],
            "frequency": one["frequency"],
        }
        normalized_iid = {
            "mean": iid["mean_edge"],
            "student_t_95_ci": iid["student_t_95_ci"],
            "positive_seeds": iid["positive_seeds"],
        }
        rows.append(
            {
                "tail": tail,
                "validated_positive_ev_state": positive_ev_gate(
                    normalized_one,
                    normalized_iid,
                    diff,
                    cidx[("one2six", tail, "neutral")],
                ),
            }
        )
    return rows


def _shape_diagnostics(rows, config):
    output = []
    for source in SOURCE_NAMES:
        slopes = []
        correlations = []
        adjacent_improvements = []
        for seed in config.validation_seeds:
            matching = sorted(
                (r for r in rows if r["source"] == source and r["seed"] == seed),
                key=lambda row: row["slice_rank"],
            )
            ranks = [float(row["slice_rank"]) for row in matching]
            edges = [float(row["player_edge_per_initial_wager"]) for row in matching]
            slopes.append(_simple_slope(ranks, edges))
            correlations.append(_correlation(ranks, edges))
            adjacent_improvements.append(
                sum(left > right for left, right in zip(edges, edges[1:], strict=False))
            )
        output.append(
            {
                "source": source,
                "direction": "negative slope means EV improves toward high-rich",
                "slope": student_t_summary(slopes),
                "spearman_style_rank_correlation": student_t_summary(correlations),
                "mean_adjacent_high_rich_improvements": _mean(adjacent_improvements),
                "maximum_possible_adjacent_improvements": len(SLICE_NAMES) - 1,
            }
        )
    return output


def _simple_slope(x_values, y_values):
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    return (
        sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)
        )
        / denominator
    )


def _correlation(x_values, y_values):
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)
    )
    denominator = sqrt(
        sum((x - x_mean) ** 2 for x in x_values)
        * sum((y - y_mean) ** 2 for y in y_values)
    )
    return numerator / denominator if denominator else 0.0


def _aggregate_insurance(rows, config):
    output = []
    for kind in ("ace_up", "even_money", "ten_up"):
        states_by_source = [
            {
                r["state"]
                for r in rows
                if r["insurance_type"] == kind and r["source"] == source
            }
            for source in SOURCE_NAMES
        ]
        states = sorted(set.intersection(*states_by_source))
        for state in states:
            source_results = {}
            for source in SOURCE_NAMES:
                matching = [
                    r
                    for r in rows
                    if r["insurance_type"] == kind
                    and r["state"] == state
                    and r["source"] == source
                ]
                probabilities = [r["event_probability"] for r in matching]
                threshold = matching[0]["break_even_probability"]
                summary = student_t_summary(probabilities)
                result = {
                    "source": source,
                    "insurance_type": kind,
                    "state": state,
                    "opportunities": sum(r["opportunities"] for r in matching),
                    "minimum_seed_opportunities": min(
                        r["opportunities"] for r in matching
                    ),
                    "mean_probability": summary["mean"],
                    "student_t_95_ci": summary["student_t_95_ci"],
                    "break_even_probability": threshold,
                    "seeds_above_threshold": sum(v > threshold for v in probabilities),
                    "implied_ev_per_unit": (3 if kind != "ten_up" else 11)
                    * summary["mean"]
                    - 1,
                    "availability_note": (
                        "rules-permitted but operational availability unconfirmed"
                    )
                    if kind == "ten_up"
                    else "formally available",
                }
                source_results[source] = result
                output.append(result)
            for source in SOURCE_NAMES:
                source_results[source]["robustly_positive"] = insurance_gate(
                    source_results[source],
                    source_results[
                        "physical_iid" if source == "one2six" else "one2six"
                    ],
                    threshold=source_results[source]["break_even_probability"],
                )
    return output


def _precision_rows(per_seed, aggregate, config):
    idx = {(r["source"], r["state"]): r for r in aggregate}
    rows = []
    for source in SOURCE_NAMES:
        for tail in HIGH_TAILS:
            values = [
                r["player_edge_per_initial_wager"]
                for r in per_seed
                if r["source"] == source and r["state"] == tail
            ]
            rows.append(
                {
                    "source": source,
                    "tail": tail,
                    "observed_round_standard_deviation": idx[(source, tail)][
                        "mean_round_standard_deviation"
                    ],
                    **planning_precision(values, idx[(source, tail)]["rounds"]),
                }
            )
    return rows


def _summary_signs(values):
    return {
        **student_t_summary(values),
        "positive_seeds": sum(v > 0 for v in values),
        "negative_seeds": sum(v < 0 for v in values),
    }


def _config_payload(config):
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["frozen_weights"] = config.weights
    return payload


def _write_outputs(config, summary, cutpoints, *tables):
    names = (
        "nested_tail_profitability.csv",
        "per_seed_nested_tail_profitability.csv",
        "disjoint_slice_profitability.csv",
        "tail_contrasts.csv",
        "paired_source_tail_differences.csv",
        "positive_ev_state_gate.csv",
        "tail_opportunity_frequency.csv",
        "tail_state_run_lengths.csv",
        "tail_precision_and_power.csv",
    )
    _write_json(config.output_dir / "summary.json", summary)
    _write_json(config.output_dir / "experiment_config.json", _config_payload(config))
    _write_json(
        config.output_dir / "extreme_tail_cutpoints.json",
        {
            "values": dict(cutpoints.values),
            "source": "one2six_development",
            "action_outcomes_used": False,
        },
    )
    for name, rows in zip(names, tables[:9], strict=True):
        _write_csv(config.output_dir / name, rows)
    insurance = tables[9]
    for kind, name in (
        ("ace_up", "ace_up_insurance_feasibility.csv"),
        ("even_money", "even_money_feasibility.csv"),
        ("ten_up", "ten_up_optional_insurance_feasibility.csv"),
    ):
        _write_csv(
            config.output_dir / name,
            [r for r in insurance if r["insurance_type"] == kind],
        )
    (config.output_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )


def _summary_markdown(summary):
    lines = [
        "This experiment tests whether the extreme favourable tail of the frozen",
        "observable fading score produces positive blackjack expected value.",
        "",
        "The main wager under the unchanged fixed strategy is the primary endpoint.",
        "",
        "Insurance and even money are secondary feasibility audits.",
        "",
        "Dealer-ten insurance is authorised by the formal rule profile but is",
        "operator-optional and is not assumed to be available in practice.",
        "",
        "No betting, playing or box-count policy is selected.",
        "",
        "# Extreme-Tail Profitability",
        "",
        "## Cutpoints and Frequencies",
        "",
        json.dumps(summary["cutpoints"]["values"], sort_keys=True),
        "",
        "## Edge by Nested High-Rich Tail",
        "",
        "| Tail | Frequency | One2Six edge | 95% CI | IID edge |",
        "|---|---:|---:|---:|---:|",
    ]
    idx = {(r["source"], r["state"]): r for r in summary["nested_tail_profitability"]}
    for tail in HIGH_TAILS:
        one = idx[("one2six", tail)]
        iid = idx[("physical_iid", tail)]
        lines.append(
            f"| {tail} | {one['frequency']:.4f} | {one['mean_edge']:.6f} | "
            f"{one['student_t_95_ci']} | {iid['mean_edge']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Positive-EV Gate Verdict",
            "",
            (
                "A tail passed."
                if any(
                    r["validated_positive_ev_state"]
                    for r in summary["positive_ev_state_gate"]
                )
                else "no validated positive-EV tail under the fixed strategy"
            ),
            "",
            "## Insurance Feasibility",
            "",
            "Exact probability thresholds and support are reported in the "
            "feasibility CSV files.",
            "",
            "## Precision and Power",
            "",
            "Planning approximations use independent-seed variance.",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend(f"- [{k}]({v})" for k, v in summary["plot_paths"].items())
    return "\n".join(lines) + "\n"


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in row.items()
                }
            )


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantile(values, q):
    return linear_quantile(values, q) if values else None


def _ci_positive(value):
    return isinstance(value, list) and len(value) == 2 and value[0] > 0


def _validate_privacy(payload):
    text = json.dumps(payload, sort_keys=True).lower()
    matches = [term for term in PRIVATE_TERMS if term in text]
    if matches:
        raise RuntimeError(f"hidden fields reached exports: {matches}")
