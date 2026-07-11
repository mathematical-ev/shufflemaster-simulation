# SPDX-License-Identifier: GPL-3.0-or-later

"""Audit monetary streak shape and incremental held-out predictive value."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Final, Literal

from experiments.fading_exclusion_validation import (
    FROZEN_WEIGHTS,
    CohortCounts,
    LedgerCardSource,
    ObservableCohortLedger,
    _play_round_capture_initial,
    calculate_fading_state,
)
from experiments.metrics import classify_monetary_outcome, streak_frequency_summary
from experiments.multi_box_counterfactual import _make_source
from experiments.observable_card_response import card_indicators
from experiments.plots import write_streak_dependence_plots
from experiments.single_box_game_validation import (
    box_round_net_result,
    student_t_summary,
)
from shufflemaster_sim.card_sources import One2SixCardSource
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.hand_values import is_natural_blackjack
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)

Source = Literal["physical_iid", "one2six"]
Outcome = Literal["win", "loss", "push"]
RunKind = Literal["win", "loss"]
ModelName = Literal["A", "B", "C"]

SOURCE_NAMES: Final[tuple[Source, ...]] = ("physical_iid", "one2six")
DEFAULT_SEEDS: Final[tuple[int, ...]] = tuple(range(72, 82))
RUN_KINDS: Final[tuple[RunKind, ...]] = ("win", "loss")
MODEL_NAMES: Final[tuple[ModelName, ...]] = ("A", "B", "C")
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
class StreakDependenceAuditConfig:
    """Frozen configuration for the monetary streak dependence audit."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    rounds_per_seed: int = 100_000
    burn_in_rounds: int = 1_000
    deck_count: int = 6
    base_bet: float = 10.0
    max_exact_streak_length: int = 20
    continuation_lengths: tuple[int, ...] = tuple(range(1, 11))
    tail_thresholds: tuple[int, ...] = (5, 8, 10, 15, 20)
    autocorrelation_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    current_rack_weight: float = 1.00
    returned_1_15_weight: float = 0.75
    returned_16_50_weight: float = 0.40
    returned_51_100_weight: float = 0.20
    returned_over_100_weight: float = 0.00
    output_dir: Path = Path("experiments/outputs/streak_dependence_audit_10x100k")

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique.")
        if self.rounds_per_seed <= 0:
            raise ValueError("rounds_per_seed must be positive.")
        if self.burn_in_rounds < 0:
            raise ValueError("burn_in_rounds must be non-negative.")
        if self.deck_count != 6:
            raise ValueError("streak audit requires exactly six decks.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        for values, name in (
            (self.continuation_lengths, "continuation lengths"),
            (self.tail_thresholds, "tail thresholds"),
            (self.autocorrelation_lags, "autocorrelation lags"),
        ):
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{name} must be positive and nonempty.")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be unique and ordered.")
        if self.max_exact_streak_length <= 0:
            raise ValueError("max_exact_streak_length must be positive.")
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
class RunRecord:
    kind: RunKind
    length: int
    left_censored: bool = False
    right_censored: bool = False


@dataclass(slots=True)
class CensoredStreakTracker:
    """Maintain live streak state and measured boundary censoring."""

    current_kind: RunKind | None = None
    current_length: int = 0
    measuring: bool = False
    current_left_censored: bool = False
    records: list[RunRecord] = field(default_factory=list)

    @property
    def sign(self) -> int:
        return 1 if self.current_kind == "win" else -1 if self.current_kind else 0

    def observe_net(self, net: float) -> None:
        self.observe(classify_monetary_outcome(net))

    def observe(self, outcome: Outcome) -> None:
        if outcome == "push":
            return
        kind: RunKind = outcome
        if self.current_kind is None:
            self.current_kind = kind
            self.current_length = 1
            self.current_left_censored = False
            return
        if self.current_kind == kind:
            self.current_length += 1
            return
        self._close(right_censored=False)
        self.current_kind = kind
        self.current_length = 1
        self.current_left_censored = False

    def start_measurement(self) -> None:
        self.records.clear()
        self.measuring = True
        self.current_left_censored = self.current_kind is not None

    def finalize_measurement(self) -> None:
        self._close(right_censored=True)
        self.measuring = False

    def _close(self, *, right_censored: bool) -> None:
        if self.measuring and self.current_kind is not None:
            self.records.append(
                RunRecord(
                    self.current_kind,
                    self.current_length,
                    self.current_left_censored,
                    right_censored,
                )
            )
        self.current_kind = None
        self.current_length = 0
        self.current_left_censored = False


@dataclass(frozen=True, slots=True)
class RoundPredictorRow:
    score: float
    streak_sign: int
    streak_length: int
    net_per_initial: float
    outcome: Outcome
    blackjack: int
    initial_hi_lo: int


@dataclass(slots=True)
class ConditionalAccumulator:
    rounds: int = 0
    score_sum: float = 0.0
    net_sum: float = 0.0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    blackjacks: int = 0
    initial_hi_lo_sum: int = 0

    def add(self, row: RoundPredictorRow) -> None:
        self.rounds += 1
        self.score_sum += row.score
        self.net_sum += row.net_per_initial
        self.wins += row.outcome == "win"
        self.losses += row.outcome == "loss"
        self.pushes += row.outcome == "push"
        self.blackjacks += row.blackjack
        self.initial_hi_lo_sum += row.initial_hi_lo

    def as_dict(self) -> dict[str, float | int]:
        n = self.rounds
        return {
            "rounds": n,
            "mean_fading_score": self.score_sum / n,
            "player_edge_per_initial_wager": self.net_sum / n,
            "win_rate": self.wins / n,
            "loss_rate": self.losses / n,
            "push_rate": self.pushes / n,
            "next_round_blackjack_rate": self.blackjacks / n,
            "next_round_initial_deal_hi_lo_mean": self.initial_hi_lo_sum / n,
        }


def geometric_pmf(continuation: float, length: int) -> float:
    """Return geometric run-length mass for a one-based run length."""
    if not 0 <= continuation <= 1 or length <= 0:
        raise ValueError("continuation must be in 0-1 and length must be positive.")
    return (1.0 - continuation) * continuation ** (length - 1)


def geometric_survival(continuation: float, length: int) -> float:
    if not 0 <= continuation <= 1 or length <= 0:
        raise ValueError("continuation must be in 0-1 and length must be positive.")
    return continuation ** (length - 1)


def geometric_shape_diagnostics(
    frequency: Mapping[int, int],
    *,
    continuation: float,
    max_exact: int,
    tail_thresholds: Sequence[int],
) -> dict[str, Any]:
    """Compare an uncensored empirical run distribution with its geometric null."""
    total = sum(frequency.values())
    bins = [*range(1, max_exact + 1), max_exact + 1]
    observed = {
        length: (
            frequency.get(length, 0)
            if length <= max_exact
            else sum(count for value, count in frequency.items() if value > max_exact)
        )
        for length in bins
    }
    expected_probability = {
        length: (
            geometric_pmf(continuation, length)
            if length <= max_exact
            else geometric_survival(continuation, length)
        )
        for length in bins
    }
    empirical_probability = {
        length: observed[length] / total if total else 0.0 for length in bins
    }
    tv = 0.5 * sum(
        abs(empirical_probability[length] - expected_probability[length])
        for length in bins
    )
    chi = sum(
        (observed[length] - total * expected_probability[length]) ** 2
        / (total * expected_probability[length])
        for length in bins
        if total * expected_probability[length] > 0
    )
    max_length = max(max(frequency, default=1), max_exact + 1)
    survival_deviation = (
        max(
            abs(
                sum(count for value, count in frequency.items() if value >= length)
                / total
                - geometric_survival(continuation, length)
            )
            for length in range(1, max_length + 1)
        )
        if total
        else None
    )
    tails = {}
    for threshold in tail_thresholds:
        empirical = (
            sum(count for value, count in frequency.items() if value >= threshold)
            / total
            if total
            else None
        )
        expected = geometric_survival(continuation, threshold)
        tails[str(threshold)] = {
            "empirical": empirical,
            "geometric": expected,
            "ratio": empirical / expected
            if empirical is not None and expected
            else None,
            "difference": empirical - expected if empirical is not None else None,
        }
    return {
        "run_count": total,
        "continuation": continuation,
        "expected_mean": 1 / (1 - continuation) if continuation < 1 else None,
        "observed": observed,
        "empirical_probability": empirical_probability,
        "expected_probability": expected_probability,
        "observed_expected_ratio": {
            length: (
                empirical_probability[length] / expected_probability[length]
                if expected_probability[length]
                else None
            )
            for length in bins
        },
        "total_variation_distance": tv,
        "chi_square_descriptive": chi,
        "maximum_survival_deviation": survival_deviation,
        "tails": tails,
    }


def continuation_diagnostics(
    records: Sequence[RunRecord], lengths: Sequence[int], *, expected: float
) -> list[dict[str, Any]]:
    """Calculate observable continuation risk sets, including censored runs."""
    rows = []
    for length in lengths:
        continued = sum(record.length >= length + 1 for record in records)
        terminated = sum(
            record.length == length and not record.right_censored for record in records
        )
        at_risk = continued + terminated
        empirical = continued / at_risk if at_risk else None
        rows.append(
            {
                "streak_length": length,
                "number_at_risk": at_risk,
                "continued": continued,
                "empirical_continuation": empirical,
                "geometric_continuation": expected,
                "difference": empirical - expected if empirical is not None else None,
                "ratio": empirical / expected
                if empirical is not None and expected
                else None,
            }
        )
    return rows


def transition_rows(
    sequence: Sequence[str], states: Sequence[str]
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter(
        zip(sequence, sequence[1:], strict=False)
    )
    rows = []
    for current in states:
        denominator = sum(counts[(current, nxt)] for nxt in states)
        for nxt in states:
            count = counts[(current, nxt)]
            rows.append(
                {
                    "current": current,
                    "next": nxt,
                    "count": count,
                    "probability": count / denominator if denominator else None,
                }
            )
    return rows


def autocorrelation(values: Sequence[float], lag: int) -> float | None:
    """Return a conventional mean-centered sample autocorrelation."""
    if lag <= 0 or len(values) <= lag:
        return None
    mean = sum(values) / len(values)
    denominator = sum((value - mean) ** 2 for value in values)
    if denominator == 0:
        return None
    numerator = sum(
        (values[index] - mean) * (values[index - lag] - mean)
        for index in range(lag, len(values))
    )
    return numerator / denominator


def model_features(row: RoundPredictorRow, model: ModelName) -> tuple[float, ...]:
    base = (1.0, row.score)
    if model == "A":
        return base
    streak = (float(row.streak_sign * row.streak_length), float(row.streak_length))
    if model == "B":
        return (*base, *streak)
    return (*base, *streak, row.score * row.streak_sign)


def fit_ols(
    rows: Sequence[RoundPredictorRow], model: ModelName, *, direction: bool = False
) -> tuple[float, ...]:
    selected = [row for row in rows if not direction or row.outcome != "push"]
    width = len(model_features(selected[0], model))
    xtx = [[0.0] * width for _ in range(width)]
    xty = [0.0] * width
    for row in selected:
        features = model_features(row, model)
        target = (
            (1.0 if row.outcome == "win" else -1.0)
            if direction
            else row.net_per_initial
        )
        for left in range(width):
            xty[left] += features[left] * target
            for right in range(width):
                xtx[left][right] += features[left] * features[right]
    solution = _solve_linear_system(xtx, xty)
    if solution is None:
        raise RuntimeError("predictive design matrix is singular.")
    return tuple(solution)


def evaluate_model(
    rows: Sequence[RoundPredictorRow],
    model: ModelName,
    coefficients: Sequence[float],
    *,
    direction: bool = False,
) -> dict[str, Any]:
    selected = [row for row in rows if not direction or row.outcome != "push"]
    predictions = [
        sum(
            coefficient * feature
            for coefficient, feature in zip(
                coefficients, model_features(row, model), strict=True
            )
        )
        for row in selected
    ]
    outcomes = [
        (1.0 if row.outcome == "win" else -1.0) if direction else row.net_per_initial
        for row in selected
    ]
    errors = [
        outcome - prediction
        for outcome, prediction in zip(outcomes, predictions, strict=True)
    ]
    return {
        "evaluation_rows": len(selected),
        "mse": sum(error * error for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "correlation": _correlation(predictions, outcomes),
        "prediction_mean": sum(predictions) / len(predictions),
        "outcome_mean": sum(outcomes) / len(outcomes),
        "sign_accuracy": (
            sum(
                (prediction >= 0) == (outcome > 0)
                for prediction, outcome in zip(predictions, outcomes, strict=True)
            )
            / len(outcomes)
            if direction
            else None
        ),
    }


def predictive_model_rows(
    observations: Sequence[RoundPredictorRow], source: Source, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    midpoint = len(observations) // 2
    estimation = observations[:midpoint]
    evaluation = observations[midpoint:]
    coefficients_rows = []
    performance_rows = []
    for target, direction in (("monetary", False), ("resolved_direction", True)):
        for model in MODEL_NAMES:
            coefficients = fit_ols(estimation, model, direction=direction)
            performance = evaluate_model(
                evaluation, model, coefficients, direction=direction
            )
            coefficients_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "target": target,
                    "model": model,
                    "estimation_rows": sum(
                        not direction or row.outcome != "push" for row in estimation
                    ),
                    "coefficients": list(coefficients),
                }
            )
            performance_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "target": target,
                    "model": model,
                    **performance,
                }
            )
    return coefficients_rows, performance_rows


def strategy_relevance_gate(
    one_summary: Mapping[str, Any],
    paired_summary: Mapping[str, Any],
    iid_summary: Mapping[str, Any],
) -> bool:
    """Apply the frozen incremental-prediction candidate gate."""
    return bool(
        float(one_summary["mean"]) > 0
        and _ci_positive(one_summary["student_t_95_ci"])
        and int(one_summary["positive_differences"]) >= 8
        and _ci_positive(paired_summary["student_t_95_ci"])
        and not (
            float(iid_summary["mean"]) > 0
            and _ci_positive(iid_summary["student_t_95_ci"])
        )
    )


def run_streak_dependence_audit(
    config: StreakDependenceAuditConfig,
) -> dict[str, Any]:
    """Run the full source-shape and held-out incremental-prediction audit."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_full: list[dict[str, Any]] = []
    distribution_binned: list[dict[str, Any]] = []
    geometric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    continuation_rows: list[dict[str, Any]] = []
    resolved_transition: list[dict[str, Any]] = []
    raw_transition: list[dict[str, Any]] = []
    autocorrelation_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    conditional: dict[tuple[str, str], ConditionalAccumulator] = defaultdict(
        ConditionalAccumulator
    )
    conditional_score: dict[tuple[str, str, str], ConditionalAccumulator] = defaultdict(
        ConditionalAccumulator
    )
    seed_metrics: list[dict[str, Any]] = []

    for source in SOURCE_NAMES:
        for seed in config.seeds:
            result = _run_seed(config, source, seed)
            seed_metrics.append(result["metrics"])
            distribution_full.extend(result["distribution_full"])
            distribution_binned.extend(result["distribution_binned"])
            geometric_rows.extend(result["geometric_rows"])
            tail_rows.extend(result["tail_rows"])
            continuation_rows.extend(result["continuation_rows"])
            resolved_transition.extend(result["resolved_transition"])
            raw_transition.extend(result["raw_transition"])
            autocorrelation_rows.extend(result["autocorrelation_rows"])
            model_rows.extend(result["model_rows"])
            performance_rows.extend(result["performance_rows"])
            for row in result["observations"]:
                group = streak_group(row.streak_sign, row.streak_length)
                conditional[(source, group)].add(row)
                conditional_score[(source, group, score_band(row.score))].add(row)

    autocorrelation_rows.extend(_aggregate_autocorrelation(autocorrelation_rows))
    shape_differences = _paired_shape_differences(seed_metrics, config)
    performance_aggregate, predictive_differences = _predictive_summaries(
        performance_rows, config
    )
    conditional_rows = [
        {"source": source, "streak_group": group, **accumulator.as_dict()}
        for (source, group), accumulator in sorted(conditional.items())
    ]
    conditional_score_rows = [
        {
            "source": source,
            "streak_group": group,
            "score_band": band,
            **accumulator.as_dict(),
        }
        for (source, group, band), accumulator in sorted(conditional_score.items())
    ]
    aggregate = _aggregate_seed_metrics(seed_metrics, config)
    serial_dependence = _serial_summary(
        resolved_transition, raw_transition, autocorrelation_rows
    )
    candidate = _candidate_verdict(performance_aggregate, predictive_differences)
    conclusion = _final_conclusion(aggregate, shape_differences, candidate)
    plot_paths = write_streak_dependence_plots(
        output_dir=config.output_dir,
        geometric_rows=geometric_rows,
        continuation_rows=continuation_rows,
        autocorrelation_rows=autocorrelation_rows,
        performance_rows=performance_rows,
        conditional_rows=conditional_rows,
        max_exact=config.max_exact_streak_length,
    )
    summary = {
        "experiment": "monetary_streak_dependence_audit",
        "config": _config_payload(config),
        "aggregate": aggregate,
        "paired_source_shape_differences": shape_differences,
        "predictive_model_performance": performance_aggregate,
        "paired_source_predictive_differences": predictive_differences,
        "serial_dependence": serial_dependence,
        "conditional_next_round": conditional_rows,
        "streak_feature_candidate": candidate,
        "conclusion": conclusion,
        "plot_paths": plot_paths,
        "hidden_state_exported": False,
    }
    _validate_privacy(summary)
    _write_outputs(
        config,
        summary,
        distribution_full,
        distribution_binned,
        geometric_rows,
        tail_rows,
        continuation_rows,
        resolved_transition,
        raw_transition,
        autocorrelation_rows,
        shape_differences,
        model_rows,
        [*performance_rows, *performance_aggregate],
        predictive_differences,
        conditional_rows,
        conditional_score_rows,
    )
    return summary


def _run_seed(
    config: StreakDependenceAuditConfig, source_name: Source, seed: int
) -> dict[str, Any]:
    source = LedgerCardSource(
        _make_source(source_name, config.deck_count, seed), ObservableCohortLedger()
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
    tracker = CensoredStreakTracker()
    round_index = 0
    for _ in range(config.burn_in_rounds):
        table = game.play_round(
            round_index=round_index, card_source=source, strategy=strategy
        )
        tracker.observe_net(box_round_net_result(table))
        round_index += 1
    tracker.start_measurement()
    observations: list[RoundPredictorRow] = []
    raw_sequence: list[str] = []
    for _ in range(config.rounds_per_seed):
        score = _prebet_score(config, source, game)
        sign = tracker.sign
        length = tracker.current_length
        table, initial = _play_round_capture_initial(
            game, source, strategy, round_index
        )
        net = box_round_net_result(table)
        outcome: Outcome = classify_monetary_outcome(net)
        blackjack = int(
            any(
                is_natural_blackjack(
                    hand.cards, blackjack_eligible=hand.blackjack_eligible
                )
                for hand in table.boxes[0].hands
            )
        )
        initial_hi_lo = sum(card_indicators(card.rank).hi_lo for card in initial)
        observations.append(
            RoundPredictorRow(
                score.predicted_hi_lo_shift,
                sign,
                length,
                net / config.base_bet,
                outcome,
                blackjack,
                initial_hi_lo,
            )
        )
        raw_sequence.append(outcome[0].upper())
        tracker.observe(outcome)
        round_index += 1
    tracker.finalize_measurement()
    if isinstance(source.source, One2SixCardSource):
        source.source.assert_invariants(game.pending_discard_rack)
    return _seed_outputs(
        config, source_name, seed, observations, raw_sequence, tracker.records
    )


def _prebet_score(
    config: StreakDependenceAuditConfig,
    source: LedgerCardSource,
    game: CasinoBlackjackGame,
):
    return calculate_fading_state(
        current_rack=CohortCounts.from_cards(game.pending_discard_rack),
        returned_by_band=source.ledger.active_by_band(source.draw_count),
        weights=config.weights,
    )


def _seed_outputs(
    config: StreakDependenceAuditConfig,
    source: Source,
    seed: int,
    observations: Sequence[RoundPredictorRow],
    raw_sequence: Sequence[str],
    records: Sequence[RunRecord],
) -> dict[str, Any]:
    resolved = [value for value in raw_sequence if value != "P"]
    wins = resolved.count("W")
    losses = resolved.count("L")
    p_win = wins / len(resolved)
    p_loss = losses / len(resolved)
    frequencies: dict[RunKind, Counter[int]] = {
        kind: Counter(
            record.length
            for record in records
            if record.kind == kind
            and not record.left_censored
            and not record.right_censored
        )
        for kind in RUN_KINDS
    }
    all_frequencies = {
        kind: Counter(record.length for record in records if record.kind == kind)
        for kind in RUN_KINDS
    }
    shapes = {
        "win": geometric_shape_diagnostics(
            frequencies["win"],
            continuation=p_win,
            max_exact=config.max_exact_streak_length,
            tail_thresholds=config.tail_thresholds,
        ),
        "loss": geometric_shape_diagnostics(
            frequencies["loss"],
            continuation=p_loss,
            max_exact=config.max_exact_streak_length,
            tail_thresholds=config.tail_thresholds,
        ),
    }
    full_rows = []
    binned_rows = []
    geometric_rows = []
    tail_rows = []
    continuation_rows = []
    for kind in RUN_KINDS:
        for scope, frequency in (
            ("primary_uncensored", frequencies[kind]),
            ("all_observed", all_frequencies[kind]),
        ):
            for length, count in sorted(frequency.items()):
                full_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "streak_type": kind,
                        "run_scope": scope,
                        "streak_length": length,
                        "frequency": count,
                    }
                )
        shape = shapes[kind]
        for length, observed in shape["observed"].items():
            label = str(length) if length <= config.max_exact_streak_length else "21+"
            binned_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "streak_type": kind,
                    "length_bin": label,
                    "frequency": observed,
                }
            )
            geometric_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "streak_type": kind,
                    "length_bin": label,
                    "observed_count": observed,
                    "expected_count": shape["expected_probability"][length]
                    * shape["run_count"],
                    "empirical_probability": shape["empirical_probability"][length],
                    "geometric_probability": shape["expected_probability"][length],
                    "observed_expected_ratio": shape["observed_expected_ratio"][length],
                    "total_variation_distance": shape["total_variation_distance"],
                    "chi_square_descriptive": shape["chi_square_descriptive"],
                    "maximum_survival_deviation": shape["maximum_survival_deviation"],
                }
            )
        for threshold, values in shape["tails"].items():
            tail_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "streak_type": kind,
                    "threshold": int(threshold),
                    **values,
                }
            )
        kind_records = [record for record in records if record.kind == kind]
        for row in continuation_diagnostics(
            kind_records,
            config.continuation_lengths,
            expected=p_win if kind == "win" else p_loss,
        ):
            continuation_rows.append(
                {"source": source, "seed": seed, "streak_type": kind, **row}
            )
        continuation_rows.extend(
            _grouped_continuation_rows(
                source,
                seed,
                kind,
                kind_records,
                expected=p_win if kind == "win" else p_loss,
            )
        )
    resolved_transition = []
    for row in transition_rows(resolved, ("W", "L")):
        benchmark = p_win if row["next"] == "W" else p_loss
        resolved_transition.append(
            {
                "source": source,
                "seed": seed,
                **row,
                "independence_probability": benchmark,
                "difference_from_independence": (
                    row["probability"] - benchmark
                    if row["probability"] is not None
                    else None
                ),
            }
        )
    raw_transition = [
        {"source": source, "seed": seed, **row}
        for row in transition_rows(raw_sequence, ("W", "P", "L"))
    ]
    autocorrelation_rows = []
    series = {
        "raw_signed": [
            1.0 if value == "W" else -1.0 if value == "L" else 0.0
            for value in raw_sequence
        ],
        "resolved_signed": [1.0 if value == "W" else -1.0 for value in resolved],
        "net_per_initial_wager": [row.net_per_initial for row in observations],
    }
    for series_name, values in series.items():
        for lag in config.autocorrelation_lags:
            autocorrelation_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "series": series_name,
                    "lag": lag,
                    "autocorrelation": autocorrelation(values, lag),
                }
            )
    model_rows, performance_rows = predictive_model_rows(observations, source, seed)
    metrics = {
        "source": source,
        "seed": seed,
        "rounds": len(observations),
        "wins": wins,
        "losses": losses,
        "pushes": len(raw_sequence) - len(resolved),
        "p_win_resolved": p_win,
        "p_loss_resolved": p_loss,
        "left_censored_runs": sum(record.left_censored for record in records),
        "right_censored_runs": sum(record.right_censored for record in records),
        "win_summary": streak_frequency_summary(frequencies["win"]),
        "loss_summary": streak_frequency_summary(frequencies["loss"]),
        "win_shape": shapes["win"],
        "loss_shape": shapes["loss"],
        "resolved_lag1": autocorrelation(series["resolved_signed"], 1),
        "raw_lag1": autocorrelation(series["raw_signed"], 1),
        "net_lag1": autocorrelation(series["net_per_initial_wager"], 1),
        "continuation": {
            kind: {
                str(row["streak_length"]): row["empirical_continuation"]
                for row in continuation_rows
                if row["streak_type"] == kind and isinstance(row["streak_length"], int)
            }
            for kind in RUN_KINDS
        },
    }
    return {
        "metrics": metrics,
        "observations": observations,
        "distribution_full": full_rows,
        "distribution_binned": binned_rows,
        "geometric_rows": geometric_rows,
        "tail_rows": tail_rows,
        "continuation_rows": continuation_rows,
        "resolved_transition": resolved_transition,
        "raw_transition": raw_transition,
        "autocorrelation_rows": autocorrelation_rows,
        "model_rows": model_rows,
        "performance_rows": performance_rows,
    }


def streak_group(sign: int, length: int) -> str:
    if sign == 0 or length == 0:
        return "none"
    prefix = "win" if sign > 0 else "loss"
    return f"{prefix}_{length}" if length < 5 else f"{prefix}_5_plus"


def score_band(score: float) -> str:
    if score < -0.0025:
        return "high-rich"
    if score > 0.0025:
        return "low-rich"
    return "neutral"


def _aggregate_seed_metrics(
    rows: Sequence[Mapping[str, Any]], config: StreakDependenceAuditConfig
) -> dict[str, Any]:
    result = {}
    for source in SOURCE_NAMES:
        matching = [row for row in rows if row["source"] == source]
        source_result = {
            "seeds": len(matching),
            "rounds": sum(int(row["rounds"]) for row in matching),
            "wins": sum(int(row["wins"]) for row in matching),
            "losses": sum(int(row["losses"]) for row in matching),
            "pushes": sum(int(row["pushes"]) for row in matching),
            "left_censored_runs": sum(
                int(row["left_censored_runs"]) for row in matching
            ),
            "right_censored_runs": sum(
                int(row["right_censored_runs"]) for row in matching
            ),
        }
        resolved = source_result["wins"] + source_result["losses"]
        source_result["p_win_resolved"] = source_result["wins"] / resolved
        source_result["p_loss_resolved"] = source_result["losses"] / resolved
        for kind in RUN_KINDS:
            pooled: Counter[int] = Counter()
            for row in matching:
                pooled.update(
                    {
                        int(length): int(count)
                        for length, count in row[f"{kind}_summary"]["frequency"].items()
                    }
                )
            source_result[f"{kind}_streaks"] = streak_frequency_summary(pooled)
            source_result[f"{kind}_total_variation"] = student_t_summary(
                [
                    float(row[f"{kind}_shape"]["total_variation_distance"])
                    for row in matching
                ]
            )
            source_result[f"{kind}_survival_deviation"] = student_t_summary(
                [
                    float(row[f"{kind}_shape"]["maximum_survival_deviation"])
                    for row in matching
                ]
            )
            source_result[f"{kind}_tail_ratios"] = {
                str(threshold): student_t_summary(
                    [
                        float(row[f"{kind}_shape"]["tails"][str(threshold)]["ratio"])
                        for row in matching
                    ]
                )
                for threshold in config.tail_thresholds
            }
            source_result[f"{kind}_continuation"] = {
                str(length): student_t_summary(
                    [
                        float(row["continuation"][kind][str(length)])
                        for row in matching
                        if row["continuation"][kind][str(length)] is not None
                    ]
                )
                for length in config.continuation_lengths
            }
        result[source] = source_result
    return result


def _paired_shape_differences(
    rows: Sequence[Mapping[str, Any]], config: StreakDependenceAuditConfig
) -> list[dict[str, Any]]:
    indexed = {(row["source"], row["seed"]): row for row in rows}
    metrics: dict[str, list[float]] = defaultdict(list)
    metric_seeds: dict[str, list[int]] = defaultdict(list)
    for seed in config.seeds:
        iid = indexed[("physical_iid", seed)]
        one = indexed[("one2six", seed)]
        for kind in RUN_KINDS:
            for field_name in (
                "total_variation_distance",
                "maximum_survival_deviation",
            ):
                name = f"{kind}_{field_name}"
                metrics[name].append(
                    float(one[f"{kind}_shape"][field_name])
                    - float(iid[f"{kind}_shape"][field_name])
                )
                metric_seeds[name].append(seed)
            for threshold in config.tail_thresholds:
                name = f"{kind}_tail_probability_ge_{threshold}"
                metrics[name].append(
                    float(one[f"{kind}_shape"]["tails"][str(threshold)]["empirical"])
                    - float(iid[f"{kind}_shape"]["tails"][str(threshold)]["empirical"])
                )
                metric_seeds[name].append(seed)
        for name in ("resolved_lag1", "raw_lag1", "net_lag1"):
            metrics[name].append(float(one[name]) - float(iid[name]))
            metric_seeds[name].append(seed)
        for kind in RUN_KINDS:
            for length in config.continuation_lengths:
                one_value = one["continuation"][kind][str(length)]
                iid_value = iid["continuation"][kind][str(length)]
                if one_value is not None and iid_value is not None:
                    metrics[f"{kind}_continuation_{length}"].append(
                        float(one_value) - float(iid_value)
                    )
                    metric_seeds[f"{kind}_continuation_{length}"].append(seed)
    return [
        _difference_row(name, values, seeds=metric_seeds[name])
        for name, values in sorted(metrics.items())
    ]


def _grouped_continuation_rows(
    source: Source,
    seed: int,
    kind: RunKind,
    records: Sequence[RunRecord],
    *,
    expected: float,
) -> list[dict[str, Any]]:
    rows = []
    for label, start, end in (("11-15", 11, 16), ("16+", 16, 17)):
        continued = sum(record.length >= end for record in records)
        terminated = sum(
            start <= record.length < end and not record.right_censored
            for record in records
        )
        at_risk = continued + terminated
        if at_risk == 0:
            continue
        geometric = expected ** (end - start)
        empirical = continued / at_risk
        rows.append(
            {
                "source": source,
                "seed": seed,
                "streak_type": kind,
                "streak_length": label,
                "number_at_risk": at_risk,
                "continued": continued,
                "empirical_continuation": empirical,
                "geometric_continuation": geometric,
                "difference": empirical - geometric,
                "ratio": empirical / geometric if geometric else None,
            }
        )
    return rows


def _aggregate_autocorrelation(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("seed") is None or row["autocorrelation"] is None:
            continue
        groups[(str(row["source"]), str(row["series"]), int(row["lag"]))].append(
            float(row["autocorrelation"])
        )
    return [
        {
            "source": source,
            "seed": None,
            "row_scope": "aggregate",
            "series": series,
            "lag": lag,
            "autocorrelation": summary["mean"],
            "student_t_95_ci": summary["student_t_95_ci"],
            "contributing_seeds": summary["independent_seed_runs"],
        }
        for (source, series, lag), values in sorted(groups.items())
        for summary in (student_t_summary(values),)
    ]


def _predictive_summaries(
    rows: Sequence[Mapping[str, Any]], config: StreakDependenceAuditConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {
        (row["source"], row["seed"], row["target"], row["model"]): row for row in rows
    }
    aggregate = []
    paired = []
    for target in ("monetary", "resolved_direction"):
        for model in ("B", "C"):
            for metric in ("mse", "mae", "correlation", "sign_accuracy"):
                if target == "monetary" and metric == "sign_accuracy":
                    continue
                if target == "resolved_direction" and metric in ("mae", "correlation"):
                    continue
                source_values = {}
                for source in SOURCE_NAMES:
                    values = []
                    for seed in config.seeds:
                        baseline = index[(source, seed, target, "A")][metric]
                        alternative = index[(source, seed, target, model)][metric]
                        if baseline is None or alternative is None:
                            continue
                        improvement = (
                            float(baseline) - float(alternative)
                            if metric in ("mse", "mae")
                            else float(alternative) - float(baseline)
                        )
                        values.append(improvement)
                    source_values[source] = values
                    aggregate.append(
                        {
                            "row_scope": "aggregate",
                            "source": source,
                            "target": target,
                            "model": model,
                            "metric": metric,
                            **_summary_with_signs(values),
                        }
                    )
                differences = [
                    one - iid
                    for one, iid in zip(
                        source_values["one2six"],
                        source_values["physical_iid"],
                        strict=True,
                    )
                ]
                paired.append(
                    {
                        "target": target,
                        "model": model,
                        "metric": metric,
                        **_summary_with_signs(differences, seeds=config.seeds),
                    }
                )
    return aggregate, paired


def _candidate_verdict(
    aggregate: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    index = {
        (row["source"], row["target"], row["model"], row["metric"]): row
        for row in aggregate
    }
    paired_index = {(row["target"], row["model"], row["metric"]): row for row in paired}
    results = {}
    for model in ("B", "C"):
        one = index[("one2six", "monetary", model, "mse")]
        iid = index[("physical_iid", "monetary", model, "mse")]
        difference = paired_index[("monetary", model, "mse")]
        results[model] = strategy_relevance_gate(one, difference, iid)
    return {
        "model_B": results["B"],
        "model_C": results["C"],
        "validated": any(results.values()),
    }


def _final_conclusion(
    aggregate: Mapping[str, Any],
    shape_differences: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> str:
    if candidate["validated"]:
        return "streak state adds validated predictive value beyond composition"
    shape_difference = any(
        _ci_excludes_zero(row["student_t_95_ci"]) for row in shape_differences
    )
    if shape_difference:
        return "streak shape differs but adds no predictive value beyond composition"
    return "streak shape appears geometric and adds no predictive value"


def _difference_row(
    name: str, values: list[float], *, seeds: Sequence[int]
) -> dict[str, Any]:
    return {"metric": name, **_summary_with_signs(values, seeds=seeds)}


def _summary_with_signs(
    values: list[float], *, seeds: Sequence[int] | None = None
) -> dict[str, Any]:
    result = {
        **student_t_summary(values),
        "positive_differences": sum(value > 0 for value in values),
        "negative_differences": sum(value < 0 for value in values),
        "zero_differences": sum(value == 0 for value in values),
    }
    if seeds is not None:
        result["differences_by_seed"] = {
            str(seed): value for seed, value in zip(seeds, values, strict=True)
        }
    return result


def _serial_summary(
    resolved_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    autocorrelation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"resolved_transition": {}, "raw_transition": {}}
    for label, rows, states in (
        ("resolved_transition", resolved_rows, ("W", "L")),
        ("raw_transition", raw_rows, ("W", "P", "L")),
    ):
        for source in SOURCE_NAMES:
            source_rows = [row for row in rows if row["source"] == source]
            result[label][source] = {}
            for current in states:
                denominator = sum(
                    int(row["count"])
                    for row in source_rows
                    if row["current"] == current
                )
                for nxt in states:
                    count = sum(
                        int(row["count"])
                        for row in source_rows
                        if row["current"] == current and row["next"] == nxt
                    )
                    result[label][source][f"{current}_to_{nxt}"] = (
                        count / denominator if denominator else None
                    )
    result["autocorrelation"] = [
        dict(row) for row in autocorrelation_rows if row.get("row_scope") == "aggregate"
    ]
    return result


def _config_payload(config: StreakDependenceAuditConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["frozen_weights"] = config.weights
    payload["score_bands"] = {
        "high-rich": "< -0.0025",
        "neutral": "[-0.0025, 0.0025]",
        "low-rich": "> 0.0025",
    }
    return payload


def _write_outputs(
    config: StreakDependenceAuditConfig,
    summary: Mapping[str, Any],
    *tables: Sequence[Mapping[str, Any]],
) -> None:
    names = (
        "streak_distribution_full.csv",
        "streak_distribution_binned.csv",
        "streak_geometric_comparison.csv",
        "streak_tail_comparison.csv",
        "streak_continuation_hazard.csv",
        "resolved_transition_matrix.csv",
        "raw_transition_matrix.csv",
        "outcome_autocorrelation.csv",
        "paired_source_shape_differences.csv",
        "per_seed_predictive_models.csv",
        "predictive_model_performance.csv",
        "paired_source_predictive_differences.csv",
        "conditional_next_round_by_streak.csv",
        "conditional_next_round_by_streak_and_score.csv",
    )
    _write_json(config.output_dir / "summary.json", summary)
    _write_json(config.output_dir / "experiment_config.json", _config_payload(config))
    for name, rows in zip(names, tables, strict=True):
        _write_csv(config.output_dir / name, rows)
    (config.output_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "This experiment audits the complete shape of monetary win and loss",
        "streaks.",
        "",
        "Previous work showed similar streak means and percentiles between",
        "Physical IID and One2Six. That did not establish geometric run lengths",
        "or outcome independence.",
        "",
        "This experiment therefore tests:",
        "",
        "1. full run-length shape against a geometric benchmark;",
        "2. source differences in continuation and tail behaviour;",
        "3. whether live streak state adds predictive value beyond the validated",
        "   fading-exclusion score.",
        "",
        "No betting or playing policy is selected.",
        "",
        "# Monetary Streak Dependence Audit",
        "",
        "## Geometric Benchmark",
        "",
    ]
    for source in SOURCE_NAMES:
        row = summary["aggregate"][source]
        lines.append(
            f"- {_source_label(source)}: resolved p(win)={row['p_win_resolved']:.6f}, "
            f"p(loss)={row['p_loss_resolved']:.6f}; implied win mean="
            f"{1 / row['p_loss_resolved']:.4f}, loss mean="
            f"{1 / row['p_win_resolved']:.4f}."
        )
    lines.extend(["", "## Full Shape Verdict", ""])
    for source in SOURCE_NAMES:
        row = summary["aggregate"][source]
        lines.append(
            f"- {_source_label(source)}: win TV="
            f"{row['win_total_variation']['mean']:.6f}, "
            f"loss TV={row['loss_total_variation']['mean']:.6f}; win survival "
            f"deviation={row['win_survival_deviation']['mean']:.6f}, loss="
            f"{row['loss_survival_deviation']['mean']:.6f}."
        )
        for kind in RUN_KINDS:
            streaks = row[f"{kind}_streaks"]
            tail_ratios = row[f"{kind}_tail_ratios"]
            continuation = row[f"{kind}_continuation"]
            lines.append(
                f"  {_source_label(source)} {kind}: runs={streaks['streak_count']}, "
                f"mean={streaks['mean']:.4f}, p95={streaks['p95']}, "
                f"tail-ratio range="
                f"{_summary_mean_range(tail_ratios)}, continuation range="
                f"{_summary_mean_range(continuation)}."
            )
    lines.extend(["", "## Source Comparison", ""])
    for row in summary["paired_source_shape_differences"]:
        lines.append(
            f"- {row['metric']}: mean={_fmt(row['mean'])}, "
            f"95% CI={_interval(row['student_t_95_ci'])}."
        )
    lines.extend(
        [
            "",
            "## Serial Dependence",
            "",
            "Resolved and raw transition matrices are reported in `summary.json` and "
            "the accompanying CSV files. Raw, resolved, and monetary "
            "autocorrelations include seed-level intervals at every configured lag.",
            "",
            "## Predictive-Value Verdict",
            "",
            f"**{summary['conclusion']}**",
            "",
            f"Model B candidate={summary['streak_feature_candidate']['model_B']}; "
            f"Model C candidate={summary['streak_feature_candidate']['model_C']}.",
            "",
            "Conditional next-round edge by live streak and fixed score band is "
            "descriptive only and is reported without cell selection.",
            "",
            "## Plots",
            "",
        ]
    )
    for label, path in summary["plot_paths"].items():
        lines.append(f"- [{label}]({path})")
    lines.append("")
    return "\n".join(lines)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float] | None:
    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot
                for value, pivot in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[index][-1] for index in range(size)]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    return covariance / sqrt(left_ss * right_ss) if left_ss and right_ss else None


def _ci_positive(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2 and value[0] > 0


def _ci_excludes_zero(value: object) -> bool:
    return (
        isinstance(value, list) and len(value) == 2 and (value[0] > 0 or value[1] < 0)
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
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _summary_mean_range(values: Mapping[str, Mapping[str, Any]]) -> str:
    means = [
        float(value["mean"]) for value in values.values() if value["mean"] is not None
    ]
    return f"[{min(means):.4f}, {max(means):.4f}]" if means else "NA"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "NA"
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _source_label(source: str) -> str:
    return "Physical IID" if source == "physical_iid" else "One2Six"
