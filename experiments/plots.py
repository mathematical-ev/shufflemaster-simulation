# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Plot helpers for experiment outputs."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from experiments.metrics import (
    gaussian_smoothed_signed_streak_density,
    geometric_probabilities,
)


def plot_interarrival_histogram(
    recurrence: Mapping[str, Any],
    *,
    probability: float,
    title: str,
    output_path: Path,
) -> Path:
    """Plot cards-between histogram with IID geometric overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    histogram = _int_key_mapping(recurrence.get("cards_between_histogram", {}))
    support = sorted(histogram) if histogram else [0]
    max_support = max(support)
    expected = geometric_probabilities(max_support, probability)
    total = sum(histogram.values()) or 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        support,
        [histogram.get(value, 0) / total for value in support],
        alpha=0.65,
        label="Observed",
    )
    ax.plot(
        list(expected),
        [expected[value] for value in expected],
        color="black",
        linewidth=1.5,
        label="IID geometric",
    )
    ax.set_title(title)
    ax.set_xlabel("Cards between appearances")
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_signed_streak_histogram(
    streaks: Mapping[str, Mapping[int, int]],
    *,
    theoretical: Mapping[str, Mapping[int, float]],
    output_path: Path,
) -> Path:
    """Plot signed win/loss streak histogram with geometric probability overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signed = _int_key_mapping(streaks.get("signed_streaks", {}))
    x_values = sorted(signed)
    total = sum(signed.values()) or 1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x_values,
        [signed[value] / total for value in x_values],
        alpha=0.65,
        label="Observed",
    )

    win_curve = _int_key_mapping(theoretical.get("win", {}))
    loss_curve = _int_key_mapping(theoretical.get("loss", {}))
    if win_curve:
        ax.plot(
            list(win_curve),
            [win_curve[value] for value in win_curve],
            color="green",
            linewidth=1.5,
            label="Win geometric",
        )
    if loss_curve:
        negative_x = [-value for value in loss_curve]
        ax.plot(
            negative_x,
            [loss_curve[value] for value in loss_curve],
            color="red",
            linewidth=1.5,
            label="Loss geometric",
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Signed Win/Loss Streak Distribution")
    ax.set_xlabel("Loss streak length < 0, win streak length > 0")
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_signed_streak_smoothed_density(
    streaks: Mapping[str, Mapping[int, int]],
    *,
    output_path: Path,
    bandwidth: float = 1.0,
) -> Path:
    """Plot a simple Gaussian-smoothed signed streak density."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signed = _int_key_mapping(streaks.get("signed_streaks", {}))
    density = gaussian_smoothed_signed_streak_density(signed, bandwidth=bandwidth)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(density), [density[value] for value in density], linewidth=1.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Signed Streak Smoothed Density")
    ax.set_xlabel("Loss streak length < 0, win streak length > 0")
    ax.set_ylabel("Smoothed density")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_outcome_percentages(
    game_metrics: Mapping[str, Any],
    *,
    output_path: Path,
) -> Path:
    """Plot key outcome rates as percentages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "Wins",
        "Losses",
        "Pushes",
        "Blackjacks",
        "Splits",
        "Doubles",
        "Busts",
    ]
    values = [
        game_metrics["win_rate_per_initial_round"],
        game_metrics["loss_rate_per_initial_round"],
        game_metrics["push_rate_per_initial_round"],
        game_metrics["player_blackjack_rate_per_initial_hand"],
        game_metrics["split_rate_per_initial_hand"],
        game_metrics["double_rate_per_initial_hand"],
        game_metrics["bust_rate_per_resolved_hand"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, [value * 100.0 for value in values], color="#4C78A8")
    ax.set_title("Outcome Percentages")
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_cumulative_profit(
    cumulative_profit_path: Sequence[float],
    *,
    output_path: Path,
) -> Path:
    """Plot cumulative profit over rounds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        range(1, len(cumulative_profit_path) + 1),
        cumulative_profit_path,
        linewidth=1.4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative Profit")
    ax.set_xlabel("Round")
    ax.set_ylabel("Net profit")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_physical_recurrence_histogram(
    recurrence: Mapping[str, Any],
    *,
    probability: float,
    title: str,
    output_path: Path,
) -> Path:
    """Plot physical cards-between recurrence counts with geometric overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    histogram = _int_key_mapping(recurrence.get("cards_between_histogram", {}))
    support = sorted(histogram) if histogram else [0]
    max_support = max(support)
    observations = int(recurrence.get("return_observations", 0))
    expected = geometric_probabilities(max_support, probability)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        support,
        [histogram.get(value, 0) for value in support],
        alpha=0.65,
        label="Observed count",
    )
    ax.plot(
        list(expected),
        [expected[value] * observations for value in expected],
        color="black",
        linewidth=1.5,
        label="IID geometric expected count",
    )
    ax.set_title(title)
    ax.set_xlabel("Cards between appearances")
    ax.set_ylabel("Return observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _int_key_mapping(mapping: Mapping[Any, Any]) -> dict[int, Any]:
    return {int(key): value for key, value in mapping.items()}
