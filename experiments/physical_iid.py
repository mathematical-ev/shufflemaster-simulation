# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Physical-card IID recurrence experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import median
from typing import Any

from experiments.metrics import geometric_probabilities
from experiments.plots import plot_physical_recurrence_histogram
from shufflemaster_sim.card_sources import PhysicalIidCardSource

DEFAULT_PHYSICAL_TEN_SPADES = "physical-iid-deck-0:T:spades"
DEFAULT_PHYSICAL_FIVE_SPADES = "physical-iid-deck-0:5:spades"
TAIL_LIMITS = (50, 100, 250, 500, 1000)
GOODNESS_OF_FIT_BINS = (
    (0, 50),
    (51, 100),
    (101, 250),
    (251, 500),
    (501, 1000),
    (1001, 2000),
    (2001, None),
)


@dataclass(frozen=True, slots=True)
class PhysicalIidRecurrenceConfig:
    """Configuration for physical-card IID recurrence experiments."""

    draws: int = 1_000_000
    deck_count: int = 6
    seed: int | None = 42
    output_dir: Path = Path("experiments/outputs/physical_iid_6deck")
    target_physical_ids: tuple[str, ...] = (
        DEFAULT_PHYSICAL_TEN_SPADES,
        DEFAULT_PHYSICAL_FIVE_SPADES,
    )

    def __post_init__(self) -> None:
        if self.draws <= 0:
            raise ValueError("draws must be positive.")
        if self.deck_count <= 0:
            raise ValueError("deck_count must be positive.")


def run_physical_iid_recurrence_experiment(
    config: PhysicalIidRecurrenceConfig,
) -> dict[str, Any]:
    """Run the physical IID recurrence experiment and write outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_physical_iid_recurrence_metrics(config)
    _write_physical_iid_plots(config, metrics)
    _write_json(config.output_dir / "metrics.json", metrics)
    return metrics


def calculate_physical_iid_recurrence_metrics(
    config: PhysicalIidRecurrenceConfig,
) -> dict[str, Any]:
    """Calculate target and pooled recurrence metrics for physical IID draws."""
    source = PhysicalIidCardSource(deck_count=config.deck_count, seed=config.seed)
    target_set = set(config.target_physical_ids)
    missing_targets = target_set - {card.physical_id for card in source.physical_cards}
    if missing_targets:
        missing = ", ".join(sorted(missing_targets))
        raise ValueError(f"Unknown target physical ids: {missing}")

    last_seen_by_physical_id: dict[str, int] = {}
    target_positions: dict[str, list[int]] = {
        target: [] for target in config.target_physical_ids
    }
    target_gaps: dict[str, list[int]] = {
        target: [] for target in config.target_physical_ids
    }
    pooled_gaps: list[int] = []
    physical_ids = [card.physical_id for card in source.physical_cards]
    appearances_by_physical_id: dict[str, int] = {
        physical_id: 0 for physical_id in physical_ids
    }
    return_counts_by_physical_id: dict[str, int] = {
        physical_id: 0 for physical_id in physical_ids
    }

    for draw_index in range(config.draws):
        card = source.draw_card()
        appearances_by_physical_id[card.physical_id] += 1
        if card.physical_id in target_set:
            target_positions[card.physical_id].append(draw_index)
        previous_draw_index = last_seen_by_physical_id.get(card.physical_id)
        if previous_draw_index is not None:
            draw_gap = draw_index - previous_draw_index
            pooled_gaps.append(draw_gap)
            return_counts_by_physical_id[card.physical_id] += 1
            if card.physical_id in target_set:
                target_gaps[card.physical_id].append(draw_gap)
        last_seen_by_physical_id[card.physical_id] = draw_index

    physical_card_count = source.physical_card_count
    probability = 1.0 / physical_card_count
    target_metrics = {
        target: recurrence_summary(
            total_draws=config.draws,
            appearances=len(target_positions[target]),
            draw_gaps=target_gaps[target],
            probability=probability,
        )
        for target in config.target_physical_ids
    }
    for target, summary in target_metrics.items():
        summary["positions"] = target_positions[target]

    pooled_recurrence = recurrence_summary(
        total_draws=config.draws,
        appearances=sum(appearances_by_physical_id.values()),
        draw_gaps=pooled_gaps,
        probability=probability,
    )
    plot_paths = _plot_paths(config)
    metrics = {
        "experiment_name": "six_deck_physical_iid_recurrence",
        "model": "physical_iid",
        "card_source_type": "PhysicalIidCardSource",
        "config": {
            "draws": config.draws,
            "deck_count": config.deck_count,
            "seed": config.seed,
            "output_dir": str(config.output_dir),
            "target_physical_ids": list(config.target_physical_ids),
        },
        "draw_index_base": 0,
        "deck_count": config.deck_count,
        "draws": config.draws,
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "physical_card_count": physical_card_count,
        "probability": probability,
        "geometric_model": (
            "For a specific labelled physical card, cards_between follows "
            "P(n) = (1 - p) ** n * p with p = 1 / physical_card_count."
        ),
        "target_physical_ids": list(config.target_physical_ids),
        "target_recurrence": target_metrics,
        "pooled_recurrence": pooled_recurrence,
        "per_physical_card_diagnostics": per_physical_card_diagnostics(
            appearance_counts=list(appearances_by_physical_id.values()),
            return_counts=list(return_counts_by_physical_id.values()),
        ),
        "plot_paths": plot_paths,
    }
    return metrics


def recurrence_summary(
    *,
    total_draws: int,
    appearances: int,
    draw_gaps: list[int],
    probability: float,
) -> dict[str, Any]:
    """Summarize draw gaps and cards-between recurrence observations."""
    cards_between = [draw_gap - 1 for draw_gap in draw_gaps]
    return_observations = len(draw_gaps)
    expected_appearances = total_draws * probability
    expected_mean_draw_gap = 1.0 / probability
    expected_mean_cards_between = (1.0 - probability) / probability
    return {
        "total_draws": total_draws,
        "appearances": appearances,
        "return_observations": return_observations,
        "expected_appearances": expected_appearances,
        "observed_appearance_rate": appearances / total_draws if total_draws else 0.0,
        "expected_appearance_rate": probability,
        "mean_draw_gap": _mean(draw_gaps),
        "expected_mean_draw_gap": expected_mean_draw_gap,
        "mean_cards_between": _mean(cards_between),
        "expected_mean_cards_between": expected_mean_cards_between,
        "median_cards_between": median(cards_between) if cards_between else None,
        "min_cards_between": min(cards_between) if cards_between else None,
        "max_cards_between": max(cards_between) if cards_between else None,
        "quantiles": _quantiles(cards_between, (0.5, 0.75, 0.9, 0.95, 0.99)),
        "tail_probabilities": {
            "observed": {
                str(limit): _tail_probability(cards_between, limit)
                for limit in TAIL_LIMITS
            },
            "theoretical": {
                str(limit): theoretical_tail_probability(probability, limit)
                for limit in TAIL_LIMITS
            },
        },
        "cards_between_histogram": dict(sorted(_histogram(cards_between).items())),
        "goodness_of_fit": binned_goodness_of_fit(
            cards_between=cards_between,
            probability=probability,
        ),
    }


def per_physical_card_diagnostics(
    *,
    appearance_counts: list[int],
    return_counts: list[int],
) -> dict[str, Any]:
    """Return compact per-card appearance and return-count diagnostics."""
    return {
        "appearance_counts": _count_distribution_summary(appearance_counts),
        "return_counts": _count_distribution_summary(return_counts),
    }


def binned_goodness_of_fit(
    *,
    cards_between: list[int],
    probability: float,
) -> dict[str, Any]:
    """Return binned observed-vs-expected geometric recurrence diagnostics."""
    total_observations = len(cards_between)
    histogram = _histogram(cards_between)
    bins = []
    chi_square_statistic = 0.0
    for lower, upper in GOODNESS_OF_FIT_BINS:
        observed = _observed_bin_count(histogram, lower, upper)
        expected_probability = _geometric_bin_probability(probability, lower, upper)
        expected = total_observations * expected_probability
        residual = observed - expected
        standardized_residual = residual / sqrt(expected) if expected > 0 else None
        if expected > 0:
            chi_square_statistic += (residual * residual) / expected
        bins.append(
            {
                "label": _bin_label(lower, upper),
                "lower": lower,
                "upper": upper,
                "observed_count": observed,
                "expected_count": expected,
                "residual": residual,
                "standardized_residual": standardized_residual,
            }
        )
    return {
        "description": (
            "Binned observed-vs-expected diagnostic under the shifted geometric "
            "IID model; chi_square_statistic is not a formal p-value."
        ),
        "bins": bins,
        "chi_square_statistic": chi_square_statistic,
    }


def _write_physical_iid_plots(
    config: PhysicalIidRecurrenceConfig,
    metrics: dict[str, Any],
) -> None:
    probability = metrics["probability"]
    for target, summary in metrics["target_recurrence"].items():
        plot_physical_recurrence_histogram(
            summary,
            probability=probability,
            title=f"Physical IID Recurrence: {target}",
            output_path=config.output_dir
            / f"target_recurrence_{_safe_name(target)}.png",
        )

    plot_physical_recurrence_histogram(
        metrics["pooled_recurrence"],
        probability=probability,
        title="Pooled Physical IID Recurrence",
        output_path=config.output_dir / "pooled_physical_recurrence.png",
    )


def _plot_paths(config: PhysicalIidRecurrenceConfig) -> dict[str, Any]:
    return {
        "target_recurrence": {
            target: f"target_recurrence_{_safe_name(target)}.png"
            for target in config.target_physical_ids
        },
        "pooled_physical_recurrence": "pooled_physical_recurrence.png",
    }


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantiles(
    values: list[int], quantiles: tuple[float, ...]
) -> dict[str, int | None]:
    if not values:
        return {f"{quantile:.2f}": None for quantile in quantiles}
    sorted_values = sorted(values)
    last_index = len(sorted_values) - 1
    return {
        f"{quantile:.2f}": sorted_values[round(last_index * quantile)]
        for quantile in quantiles
    }


def _tail_probability(values: list[int], limit: int) -> float:
    return sum(1 for value in values if value <= limit) / len(values) if values else 0.0


def theoretical_tail_probability(probability: float, limit: int) -> float:
    """Return P(cards_between <= limit) under shifted geometric IID."""
    return 1.0 - ((1.0 - probability) ** (limit + 1))


def _histogram(values: list[int]) -> dict[int, int]:
    histogram: defaultdict[int, int] = defaultdict(int)
    for value in values:
        histogram[value] += 1
    return dict(histogram)


def _count_distribution_summary(values: list[int]) -> dict[str, Any]:
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": _mean(values),
        "median": median(values) if values else None,
        "quantiles": _quantiles(values, (0.05, 0.25, 0.5, 0.75, 0.95)),
        "distribution": dict(sorted(_histogram(values).items())),
    }


def _observed_bin_count(
    histogram: dict[int, int],
    lower: int,
    upper: int | None,
) -> int:
    if upper is None:
        return sum(count for value, count in histogram.items() if value >= lower)
    return sum(count for value, count in histogram.items() if lower <= value <= upper)


def _geometric_bin_probability(
    probability: float,
    lower: int,
    upper: int | None,
) -> float:
    lower_tail_before = 1.0 - ((1.0 - probability) ** lower)
    if upper is None:
        return 1.0 - lower_tail_before
    upper_tail = theoretical_tail_probability(probability, upper)
    return upper_tail - lower_tail_before


def _bin_label(lower: int, upper: int | None) -> str:
    return f"{lower}+" if upper is None else f"{lower}-{upper}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_name(value: str) -> str:
    return value.replace(":", "_").replace("-", "_")


def expected_cards_between_probabilities(
    *,
    physical_card_count: int,
    max_cards_between: int,
) -> dict[int, float]:
    """Return theoretical cards-between probabilities for physical IID."""
    return geometric_probabilities(max_cards_between, 1.0 / physical_card_count)
