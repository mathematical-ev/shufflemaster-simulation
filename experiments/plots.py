# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot helpers for experiment outputs."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isnan
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiments.metrics import (
    gaussian_smoothed_signed_streak_density,
    geometric_probabilities,
    signed_streak_histogram_data,
)


def plot_counterfactual_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_keys: Sequence[str],
    column_keys: Sequence[int],
    row_field: str,
    column_field: str,
    value_field: str,
    sparse_field: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    percentage: bool = False,
) -> Path:
    """Plot an annotated state-action matrix and mark sparse cells."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indexed = {(row[row_field], int(row[column_field])): row for row in rows}
    matrix: list[list[float]] = []
    annotations: list[list[str]] = []
    finite_values: list[float] = []
    non_sparse_values: list[float] = []
    for row_key in row_keys:
        matrix_row: list[float] = []
        annotation_row: list[str] = []
        for column_key in column_keys:
            row = indexed.get((row_key, column_key))
            value = row.get(value_field) if row is not None else None
            sparse = bool(row is None or row.get(sparse_field, False))
            numeric = float("nan") if value is None else float(value)
            matrix_row.append(numeric)
            if not isnan(numeric):
                finite_values.append(numeric)
                if not sparse:
                    non_sparse_values.append(numeric)
                formatted = f"{numeric:.2%}" if percentage else f"{numeric:.3f}"
            else:
                formatted = "n/a"
            annotation_row.append(f"{formatted}{'*' if sparse else ''}")
        matrix.append(matrix_row)
        annotations.append(annotation_row)

    scale_values = non_sparse_values or finite_values
    scale = max((abs(value) for value in scale_values), default=1.0)
    fig_width = max(7.5, 1.1 * len(column_keys) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-scale,
        vmax=scale,
        aspect="auto",
    )
    ax.set_xticks(range(len(column_keys)), [str(value) for value in column_keys])
    ax.set_yticks(
        range(len(row_keys)),
        [value.replace("_", " ").title() for value in row_keys],
    )
    for row_index, annotation_row in enumerate(annotations):
        for column_index, annotation in enumerate(annotation_row):
            value = matrix[row_index][column_index]
            color = (
                "white" if not isnan(value) and abs(value) > scale * 0.55 else "black"
            )
            ax.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(image, ax=ax, label="Estimate")
    fig.text(0.5, 0.01, "* fewer than 100 sampled states", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_response_slopes(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    x_values: Sequence[int | str],
    title: str,
    x_label: str,
    output_path: Path,
    reference_one: bool = True,
) -> Path:
    """Plot seed-level response slopes and confidence intervals by position/lag."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    positions = list(range(len(x_values)))
    for source_name, offset in (("physical_iid", -0.08), ("one2six", 0.08)):
        indexed = {
            str(row[x_field]): row for row in rows if row["source"] == source_name
        }
        means: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        valid_positions: list[float] = []
        for index, value in enumerate(x_values):
            row = indexed.get(str(value))
            if row is None or row.get("mean_seed_slope") is None:
                continue
            mean = float(row["mean_seed_slope"])
            interval = row.get("student_t_95_ci")
            if isinstance(interval, list) and len(interval) == 2:
                lower, upper = float(interval[0]), float(interval[1])
            else:
                lower = upper = mean
            valid_positions.append(index + offset)
            means.append(mean)
            lower_errors.append(mean - lower)
            upper_errors.append(upper - mean)
        ax.errorbar(
            valid_positions,
            means,
            yerr=[lower_errors, upper_errors],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            capsize=2,
            label=_source_display_name(source_name),
            color=colors[source_name],
        )
    ax.axhline(0, color="black", linewidth=0.9)
    if reference_one:
        ax.axhline(1, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions, [str(value) for value in x_values], rotation=35)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean seed-level response slope")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_memory_horizon(
    band_rows: Sequence[Mapping[str, Any]],
    *,
    memory_rows: Sequence[Mapping[str, Any]],
    lag_labels: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot One2Six lag-band response slopes for each composition outcome."""
    _ = memory_rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(lag_labels)
    positions = list(range(len(labels)))
    colors = {
        "hi_lo": "#222222",
        "low": "#59A14F",
        "neutral": "#9C755F",
        "ten_value": "#E15759",
        "ace": "#4E79A7",
    }
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for outcome, color in colors.items():
        indexed = {
            str(row["lag"]): row
            for row in band_rows
            if row["source"] == "one2six" and row["outcome"] == outcome
        }
        values = [indexed[label].get("mean_seed_slope") for label in labels]
        ax.plot(
            positions,
            values,
            marker="o",
            linewidth=1.5,
            label=outcome.replace("_", " "),
            color=color,
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axhline(0.10, color="#777777", linewidth=0.8, linestyle="--")
    ax.axhline(-0.10, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions, labels)
    ax.set_title("One2Six observable returned-cohort response by lag band")
    ax.set_xlabel("Future dealt-card lag band")
    ax.set_ylabel("Mean seed-level response slope")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_heldout_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    output_path: Path,
) -> Path:
    """Plot held-out source slopes with seed-level confidence intervals."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Physical IID", "One2Six"]
    sources = ["physical_iid", "one2six"]
    means, lower, upper = [], [], []
    for source in sources:
        row = next(item for item in rows if item["source"] == source)
        mean = float(row["mean_seed_slope"])
        interval = row["student_t_95_ci"]
        means.append(mean)
        if isinstance(interval, list) and len(interval) == 2:
            lower.append(mean - float(interval[0]))
            upper.append(float(interval[1]) - mean)
        else:
            lower.append(0.0)
            upper.append(0.0)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.errorbar(
        range(2),
        means,
        yerr=[lower, upper],
        fmt="o",
        capsize=5,
        color="#4C78A8",
    )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axhline(1, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(2), labels)
    ax.set_title(title)
    ax.set_ylabel("Mean held-out seed slope")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_score_distribution(
    score_values: Mapping[str, Sequence[float]], *, output_path: Path
) -> Path:
    """Plot held-out frozen Hi-Lo score distributions by source."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for source, color in (("physical_iid", "#4C78A8"), ("one2six", "#E45756")):
        ax.hist(
            score_values[source],
            bins=60,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=_source_display_name(source),
            color=color,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Held-out frozen fading-score distribution")
    ax.set_xlabel("Predicted Hi-Lo shift")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_component_contributions(
    rows: Sequence[Mapping[str, Any]], *, output_path: Path
) -> Path:
    """Plot Hi-Lo component variability for each source."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    components = [
        "current_rack",
        "returned_1_15",
        "returned_16_50",
        "returned_51_100",
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.36
    for source_index, source in enumerate(("physical_iid", "one2six")):
        row = next(
            item
            for item in rows
            if item["source"] == source and item["outcome"] == "hi_lo"
        )
        offset = (source_index - 0.5) * width
        ax.bar(
            [index + offset for index in range(len(components))],
            [row[f"sd_{component}"] for component in components],
            width=width,
            label=_source_display_name(source),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(
        range(len(components)),
        [component.replace("_", " ") for component in components],
        rotation=20,
    )
    ax.set_title("Variability of weighted Hi-Lo cohort contributions")
    ax.set_ylabel("Contribution standard deviation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_score_group_edges(
    rows: Sequence[Mapping[str, Any]], *, output_path: Path
) -> Path:
    """Plot exploratory actual-round edge by fixed score group."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = ["predicted_high_rich", "near_neutral", "predicted_low_rich"]
    aggregate = [row for row in rows if row["seed_or_aggregate"] == "aggregate"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.36
    for source_index, source in enumerate(("physical_iid", "one2six")):
        indexed = {
            row["score_group"]: row for row in aggregate if row["source"] == source
        }
        offset = (source_index - 0.5) * width
        means = []
        for group in groups:
            mean = indexed[group].get("mean_seed_edge_per_initial_wager")
            means.append(
                indexed[group]["edge_per_initial_wager"] if mean is None else mean
            )
        lower_errors = []
        upper_errors = []
        for group, mean in zip(groups, means, strict=True):
            interval = indexed[group].get("seed_edge_student_t_95_ci")
            if isinstance(interval, list) and len(interval) == 2:
                lower_errors.append(float(mean) - float(interval[0]))
                upper_errors.append(float(interval[1]) - float(mean))
            else:
                lower_errors.append(0.0)
                upper_errors.append(0.0)
        ax.bar(
            [index + offset for index in range(len(groups))],
            means,
            width=width,
            yerr=[lower_errors, upper_errors],
            capsize=3,
            label=_source_display_name(source),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(groups)), [group.replace("_", " ") for group in groups])
    ax.set_title("Exploratory actual-round edge by frozen score group")
    ax.set_ylabel("Player edge per initial wager")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


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


def plot_comparative_signed_streak_histogram(
    source_streaks: Mapping[str, Mapping[str, Any]],
    *,
    output_path: Path,
    display_limit: int = 20,
) -> Path:
    """Plot normalized signed monetary streak lengths for multiple sources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_data: dict[str, dict[str, Any]] = {}
    for source_name, streaks in source_streaks.items():
        source_data[source_name] = signed_streak_histogram_data(
            _int_key_mapping(streaks["win_streaks"]["frequency"]),
            _int_key_mapping(streaks["loss_streaks"]["frequency"]),
            display_limit=display_limit,
        )

    x_values = [
        value for value in range(-display_limit - 1, display_limit + 2) if value != 0
    ]
    source_names = list(source_data)
    width = 0.8 / max(len(source_names), 1)
    colors = ("#4C78A8", "#E45756")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, source_name in enumerate(source_names):
        offset = (index - (len(source_names) - 1) / 2) * width
        proportions = source_data[source_name]["proportions"]
        ax.bar(
            [value + offset for value in x_values],
            [proportions[value] for value in x_values],
            width=width,
            alpha=0.78,
            label=_source_display_name(source_name),
            color=colors[index % len(colors)],
        )

    tick_values = [
        -display_limit - 1,
        -15,
        -10,
        -5,
        -1,
        1,
        5,
        10,
        15,
        display_limit + 1,
    ]
    tick_values = sorted(set(value for value in tick_values if value in x_values))
    tick_labels = [
        f"<=-{display_limit + 1}"
        if value == -display_limit - 1
        else f">=+{display_limit + 1}"
        if value == display_limit + 1
        else str(value)
        for value in tick_values
    ]
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xticks(tick_values, tick_labels)
    ax.set_title("Signed Monetary Streak Lengths (loss < 0, win > 0)")
    ax.set_xlabel("Signed streak length")
    ax.set_ylabel("Proportion of streaks")
    ax.legend()
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


def _source_display_name(source_name: str) -> str:
    if source_name == "physical_iid":
        return "Physical IID"
    if source_name == "one2six":
        return "One2Six"
    return source_name


CONDITIONAL_SCORE_BANDS = (
    "strong_high_rich",
    "moderate_high_rich",
    "neutral",
    "moderate_low_rich",
    "strong_low_rich",
)


def plot_conditional_edge_by_band(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str | None,
    output_path: Path,
) -> Path:
    """Plot player edge and seed-level intervals across frozen score bands."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sources = (source,) if source is not None else ("physical_iid", "one2six")
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    width = 0.36 if len(sources) == 2 else 0.55
    for source_index, source_name in enumerate(sources):
        indexed = {
            row["score_band"]: row for row in rows if row["source"] == source_name
        }
        means = [
            float(indexed[band]["mean_seed_player_edge_per_initial_wager"])
            for band in CONDITIONAL_SCORE_BANDS
        ]
        errors = [
            _asymmetric_errors(
                mean,
                indexed[band]["player_edge_per_initial_wager_seed_ci"],
            )
            for mean, band in zip(means, CONDITIONAL_SCORE_BANDS, strict=True)
        ]
        offset = (source_index - (len(sources) - 1) / 2) * width
        ax.bar(
            [index + offset for index in range(len(CONDITIONAL_SCORE_BANDS))],
            means,
            width=width,
            yerr=[
                [error[0] for error in errors],
                [error[1] for error in errors],
            ],
            capsize=3,
            label=_source_display_name(source_name),
            color=colors[source_name],
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(
        range(len(CONDITIONAL_SCORE_BANDS)),
        [band.replace("_", " ") for band in CONDITIONAL_SCORE_BANDS],
        rotation=18,
    )
    ax.set_title("Held-out player edge by frozen score band")
    ax.set_xlabel("Predicted high-rich <- frozen score band -> predicted low-rich")
    ax.set_ylabel("Player edge per initial wager")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_monetary_slopes_by_seed(
    per_seed: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
) -> Path:
    """Plot each validation-seed monetary slope and aggregate uncertainty."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    seeds = sorted({int(row["seed"]) for row in per_seed})
    for source_name, offset in (("physical_iid", -0.08), ("one2six", 0.08)):
        source_rows = {
            int(row["seed"]): row for row in per_seed if row["source"] == source_name
        }
        ax.scatter(
            [seed + offset for seed in seeds],
            [source_rows[seed]["slope"] for seed in seeds],
            label=_source_display_name(source_name),
            color=colors[source_name],
            alpha=0.8,
        )
        aggregate_row = next(row for row in aggregate if row["source"] == source_name)
        mean = float(aggregate_row["mean_monetary_slope"])
        interval = aggregate_row["student_t_95_ci"]
        ax.axhline(mean, color=colors[source_name], linewidth=1.2)
        if isinstance(interval, list) and len(interval) == 2:
            ax.axhspan(
                float(interval[0]),
                float(interval[1]),
                color=colors[source_name],
                alpha=0.08,
            )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(seeds)
    ax.set_title("Held-out continuous monetary slope by independent seed")
    ax.set_xlabel("Validation seed")
    ax.set_ylabel("Net per initial wager / predicted Hi-Lo shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_actual_vs_placebo_slopes(
    rows: Sequence[Mapping[str, Any]], *, output_path: Path
) -> Path:
    """Plot actual and deterministic-placebo slopes for every seed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharey=True)
    for ax, source_name in zip(axes, ("physical_iid", "one2six"), strict=True):
        matching = sorted(
            (row for row in rows if row["source"] == source_name),
            key=lambda row: int(row["seed"]),
        )
        for row in matching:
            ax.plot(
                [0, 1],
                [row["actual_slope"], row["placebo_slope"]],
                marker="o",
                linewidth=0.8,
                alpha=0.6,
            )
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks([0, 1], ["Actual", "Permutation placebo"])
        ax.set_title(_source_display_name(source_name))
    axes[0].set_ylabel("Monetary slope")
    fig.suptitle("Actual versus deterministic within-seed placebo slopes")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_initial_composition_by_band(
    rows: Sequence[Mapping[str, Any]], *, output_path: Path
) -> Path:
    """Plot combined initial-deal Hi-Lo mean with seed-level intervals."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    positions = list(range(len(CONDITIONAL_SCORE_BANDS)))
    for source_name in ("physical_iid", "one2six"):
        indexed = {
            row["score_band"]: row for row in rows if row["source"] == source_name
        }
        means = [
            float(indexed[band]["mean_seed_combined_initial_hi_lo_mean"])
            for band in CONDITIONAL_SCORE_BANDS
        ]
        errors = [
            _asymmetric_errors(
                mean,
                indexed[band]["combined_initial_hi_lo_mean_seed_ci"],
            )
            for mean, band in zip(means, CONDITIONAL_SCORE_BANDS, strict=True)
        ]
        ax.errorbar(
            positions,
            means,
            yerr=[
                [error[0] for error in errors],
                [error[1] for error in errors],
            ],
            marker="o",
            capsize=3,
            label=_source_display_name(source_name),
            color=colors[source_name],
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(
        positions,
        [band.replace("_", " ") for band in CONDITIONAL_SCORE_BANDS],
        rotation=18,
    )
    ax.set_title("Initial-deal composition by frozen score band")
    ax.set_xlabel("Predicted high-rich <- frozen score band -> predicted low-rich")
    ax.set_ylabel("Combined initial-deal mean Hi-Lo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_score_band_frequency(
    rows: Sequence[Mapping[str, Any]], *, output_path: Path
) -> Path:
    """Plot held-out opportunity frequency and seed-level uncertainty."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    width = 0.36
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for source_index, source_name in enumerate(("physical_iid", "one2six")):
        indexed = {
            row["score_band"]: row for row in rows if row["source"] == source_name
        }
        means = [
            float(indexed[band]["mean_seed_round_frequency"])
            for band in CONDITIONAL_SCORE_BANDS
        ]
        errors = [
            _asymmetric_errors(mean, indexed[band]["round_frequency_seed_ci"])
            for mean, band in zip(means, CONDITIONAL_SCORE_BANDS, strict=True)
        ]
        offset = (source_index - 0.5) * width
        ax.bar(
            [index + offset for index in range(len(CONDITIONAL_SCORE_BANDS))],
            means,
            width=width,
            yerr=[
                [error[0] for error in errors],
                [error[1] for error in errors],
            ],
            capsize=3,
            color=colors[source_name],
            label=_source_display_name(source_name),
        )
    ax.set_xticks(
        range(len(CONDITIONAL_SCORE_BANDS)),
        [band.replace("_", " ") for band in CONDITIONAL_SCORE_BANDS],
        rotation=18,
    )
    ax.set_title("Held-out score-band opportunity frequency")
    ax.set_xlabel("Predicted high-rich <- frozen score band -> predicted low-rich")
    ax.set_ylabel("Fraction of validation rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _asymmetric_errors(mean: float, interval: object) -> tuple[float, float]:
    if isinstance(interval, list) and len(interval) == 2:
        return mean - float(interval[0]), float(interval[1]) - mean
    return 0.0, 0.0


def plot_action_delta_by_cell(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_support: int,
    output_path: Path,
) -> Path:
    """Plot the most-supported development ten/ace action deltas."""
    selected = _most_supported_action_rows(rows, minimum_support)
    return _plot_action_delta_rows(
        selected,
        title="Development action deltas by ten-value/ace cell",
        output_path=output_path,
    )


def plot_action_delta_by_low_band(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_support: int,
    output_path: Path,
) -> Path:
    """Plot the most-supported development low-band action deltas."""
    selected = _most_supported_action_rows(rows, minimum_support)
    return _plot_action_delta_rows(
        selected,
        title="Development action deltas by low-card band",
        output_path=output_path,
    )


def _most_supported_action_rows(
    rows: Sequence[Mapping[str, Any]], minimum_support: int
) -> list[Mapping[str, Any]]:
    supported = [
        row for row in rows if int(row.get("sampled_states", 0)) >= minimum_support
    ]
    return sorted(
        supported,
        key=lambda row: (-int(row["sampled_states"]), str(row["decision_id"])),
    )[:20]


def _plot_action_delta_rows(
    rows: Sequence[Mapping[str, Any]], *, title: str, output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    if not rows:
        ax.text(0.5, 0.5, "No sufficiently supported states", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [_action_plot_label(row) for row in rows]
        means = [float(row["mean_seed_delta_vs_baseline"]) for row in rows]
        errors = [
            _asymmetric_errors(mean, row["delta_student_t_95_ci"])
            for mean, row in zip(means, rows, strict=True)
        ]
        positions = list(range(len(rows)))
        colors = [
            "#E45756" if row["source"] == "one2six" else "#4C78A8" for row in rows
        ]
        ax.barh(
            positions,
            means,
            xerr=[
                [error[0] for error in errors],
                [error[1] for error in errors],
            ],
            color=colors,
            capsize=2,
        )
        ax.set_yticks(positions, labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Mean seed paired delta versus baseline")
        ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_ace_vs_ten_response(
    rows: Sequence[Mapping[str, Any]], *, output_path: Path
) -> Path:
    """Plot separate ace and ten-value continuous action-response coefficients."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(
        rows,
        key=lambda row: (
            -int(row.get("sampled_states", 0)),
            str(row.get("decision_id")),
        ),
    )[:40]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    markers = {"development": "o", "validation": "s"}
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for row in selected:
        ax.scatter(
            row["mean_ten_value_coefficient"],
            row["mean_ace_coefficient"],
            marker=markers[str(row["phase"])],
            color=colors[str(row["source"])],
            alpha=0.7,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Ace versus ten-value action-response coefficients")
    ax.set_xlabel("Ten-value coefficient")
    ax.set_ylabel("Ace coefficient")
    legend_items = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#666666",
            label=phase.title(),
            markersize=7,
        )
        for phase, marker in markers.items()
    ]
    legend_items.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            label="Physical IID" if source == "physical_iid" else "One2Six",
            markersize=7,
        )
        for source, color in colors.items()
    )
    ax.legend(handles=legend_items, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_validated_candidate_deltas(
    validation_rows: Sequence[Mapping[str, Any]],
    validated_rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
) -> Path:
    """Plot frozen held-out candidates, highlighting validated classifications."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        validation_rows,
        key=lambda row: (
            -int(row.get("validation_sampled_states", 0)),
            str(row["candidate_id"]),
        ),
    )[:25]
    validated_ids = {row["candidate_id"] for row in validated_rows}
    fig, ax = plt.subplots(figsize=(11, 7))
    if not rows:
        ax.text(0.5, 0.5, "No frozen development candidates", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [_candidate_plot_label(row) for row in rows]
        means = [
            float(row["validation_mean_delta"])
            if row["validation_mean_delta"] is not None
            else 0.0
            for row in rows
        ]
        errors = [
            _asymmetric_errors(mean, row["validation_delta_ci"])
            for mean, row in zip(means, rows, strict=True)
        ]
        positions = list(range(len(rows)))
        ax.barh(
            positions,
            means,
            xerr=[
                [error[0] for error in errors],
                [error[1] for error in errors],
            ],
            color=[
                "#59A14F" if row["candidate_id"] in validated_ids else "#9C9C9C"
                for row in rows
            ],
            capsize=2,
        )
        ax.set_yticks(positions, labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Held-out paired delta versus baseline")
        ax.invert_yaxis()
    ax.set_title("Frozen candidate held-out action deltas")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_decision_state_support(
    rows: Sequence[Mapping[str, Any]], *, output_path: Path
) -> Path:
    """Plot the most frequent exact observable decision keys."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted(rows, key=lambda row: -int(row["decision_count"]))[:25]
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [_support_plot_label(row) for row in selected]
    positions = list(range(len(selected)))
    ax.barh(
        positions,
        [int(row["decision_count"]) for row in selected],
        color=[
            "#E45756" if row["source"] == "one2six" else "#4C78A8" for row in selected
        ],
    )
    ax.set_yticks(positions, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Most frequent exact decision states")
    ax.set_xlabel("Decision snapshots")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _action_plot_label(row: Mapping[str, Any]) -> str:
    return (
        f"{row['source']} {row['hand_kind']}{row['hand_total']} v "
        f"{row['dealer_upcard']} {row['composition_state']} -> {row['action']}"
    )


def _candidate_plot_label(row: Mapping[str, Any]) -> str:
    return (
        f"{row['hand_kind']}{row['hand_total']} v {row['dealer_upcard']} "
        f"{row['composition_state']}: {row['baseline_action']}->"
        f"{row['alternative_action']}"
    )


def _support_plot_label(row: Mapping[str, Any]) -> str:
    return (
        f"{row['source']} {row['phase']} {row['hand_kind']}{row['hand_total']} "
        f"v {row['dealer_upcard']} {row['baseline_action']}"
    )


def write_streak_dependence_plots(
    *,
    output_dir: Path,
    geometric_rows: Sequence[Mapping[str, Any]],
    continuation_rows: Sequence[Mapping[str, Any]],
    autocorrelation_rows: Sequence[Mapping[str, Any]],
    performance_rows: Sequence[Mapping[str, Any]],
    conditional_rows: Sequence[Mapping[str, Any]],
    max_exact: int,
) -> dict[str, str]:
    """Write the compact monetary streak-shape audit plot set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for kind in ("win", "loss"):
        name = f"{kind}_streak_pmf_vs_geometric.png"
        _plot_streak_pmf(geometric_rows, kind, output_dir / name, max_exact)
        paths[f"{kind}_streak_pmf_vs_geometric"] = name
        name = f"{kind}_streak_survival_vs_geometric.png"
        _plot_streak_survival(geometric_rows, kind, output_dir / name, max_exact)
        paths[f"{kind}_streak_survival_vs_geometric"] = name
        name = f"{kind}_streak_continuation_probability.png"
        _plot_streak_continuation(continuation_rows, kind, output_dir / name)
        paths[f"{kind}_streak_continuation_probability"] = name
    name = "observed_expected_streak_ratio.png"
    _plot_observed_expected_ratio(geometric_rows, output_dir / name, max_exact)
    paths["observed_expected_streak_ratio"] = name
    name = "outcome_autocorrelation_by_lag.png"
    _plot_outcome_autocorrelation(autocorrelation_rows, output_dir / name)
    paths["outcome_autocorrelation_by_lag"] = name
    name = "predictive_mse_improvement_by_seed.png"
    _plot_predictive_improvement(performance_rows, output_dir / name)
    paths["predictive_mse_improvement_by_seed"] = name
    name = "conditional_edge_by_signed_streak.png"
    _plot_conditional_streak_edge(conditional_rows, output_dir / name)
    paths["conditional_edge_by_signed_streak"] = name
    return paths


def _streak_plot_groups(
    rows: Sequence[Mapping[str, Any]], kind: str, max_exact: int
) -> dict[str, dict[int, tuple[float, float]]]:
    values: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for row in rows:
        if row["streak_type"] != kind or row["length_bin"] == f"{max_exact + 1}+":
            continue
        try:
            length = int(row["length_bin"])
        except ValueError:
            continue
        values.setdefault((str(row["source"]), length), []).append(
            (float(row["empirical_probability"]), float(row["geometric_probability"]))
        )
    grouped: dict[str, dict[int, tuple[float, float]]] = {}
    for (source, length), pairs in values.items():
        grouped.setdefault(source, {})[length] = (
            sum(pair[0] for pair in pairs) / len(pairs),
            sum(pair[1] for pair in pairs) / len(pairs),
        )
    return grouped


def _plot_streak_pmf(
    rows: Sequence[Mapping[str, Any]], kind: str, path: Path, max_exact: int
) -> None:
    grouped = _streak_plot_groups(rows, kind, max_exact)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for source, by_length in grouped.items():
        lengths = sorted(by_length)
        ax.plot(
            lengths,
            [by_length[length][0] for length in lengths],
            marker="o",
            markersize=3,
            color=colors[source],
            label=f"{source} empirical",
        )
        ax.plot(
            lengths,
            [by_length[length][1] for length in lengths],
            linestyle="--",
            color=colors[source],
            label=f"{source} geometric",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Streak length")
    ax.set_ylabel("Probability mass (log scale)")
    ax.set_title(f"{kind.title()} streak PMF versus geometric benchmark")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_streak_survival(
    rows: Sequence[Mapping[str, Any]], kind: str, path: Path, max_exact: int
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    lengths = list(range(1, max_exact + 1))
    for source in colors:
        source_rows = [
            row
            for row in rows
            if row["source"] == source and row["streak_type"] == kind
        ]
        seeds = sorted({int(row["seed"]) for row in source_rows})
        empirical = []
        geometric = []
        for length in lengths:
            empirical_by_seed = []
            geometric_by_seed = []
            for seed in seeds:
                matching = [row for row in source_rows if int(row["seed"]) == seed]
                empirical_by_seed.append(
                    sum(
                        float(row["empirical_probability"])
                        for row in matching
                        if row["length_bin"] == f"{max_exact + 1}+"
                        or int(row["length_bin"]) >= length
                    )
                )
                geometric_by_seed.append(
                    sum(
                        float(row["geometric_probability"])
                        for row in matching
                        if row["length_bin"] == f"{max_exact + 1}+"
                        or int(row["length_bin"]) >= length
                    )
                )
            empirical.append(sum(empirical_by_seed) / len(empirical_by_seed))
            geometric.append(sum(geometric_by_seed) / len(geometric_by_seed))
        ax.plot(lengths, empirical, color=colors[source], label=f"{source} empirical")
        ax.plot(
            lengths,
            geometric,
            color=colors[source],
            linestyle="--",
            label=f"{source} geometric",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Streak length at least k")
    ax.set_ylabel("Survival probability (log scale)")
    ax.set_title(f"{kind.title()} streak survival versus geometric benchmark")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_streak_continuation(
    rows: Sequence[Mapping[str, Any]], kind: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for source in colors:
        selected = [
            row
            for row in rows
            if row["source"] == source
            and row["streak_type"] == kind
            and isinstance(row["streak_length"], int)
        ]
        lengths = sorted({int(row["streak_length"]) for row in selected})
        means = []
        errors = []
        expected = []
        supported_lengths = []
        for length in lengths:
            matching = [row for row in selected if row["streak_length"] == length]
            values = [
                float(row["empirical_continuation"])
                for row in matching
                if row["empirical_continuation"] is not None
            ]
            if not values:
                continue
            supported_lengths.append(length)
            mean = sum(values) / len(values)
            standard_error = (
                (sum((value - mean) ** 2 for value in values) / (len(values) - 1))
                ** 0.5
                / len(values) ** 0.5
                if len(values) > 1
                else 0.0
            )
            means.append(mean)
            errors.append(1.96 * standard_error)
            expected.append(
                sum(float(row["geometric_continuation"]) for row in matching)
                / len(matching)
            )
        ax.errorbar(
            supported_lengths,
            means,
            yerr=errors,
            marker="o",
            color=colors[source],
            label=f"{source} empirical",
        )
        ax.plot(
            supported_lengths,
            expected,
            linestyle="--",
            color=colors[source],
            label=f"{source} geometric",
        )
    ax.set_xlabel("Current streak length")
    ax.set_ylabel("Continuation probability")
    ax.set_title(f"{kind.title()} streak continuation with seed uncertainty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_observed_expected_ratio(
    rows: Sequence[Mapping[str, Any]], path: Path, max_exact: int
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        ("physical_iid", "win"): "#4C78A8",
        ("physical_iid", "loss"): "#72A0C1",
        ("one2six", "win"): "#E45756",
        ("one2six", "loss"): "#F28E8B",
    }
    for (source, kind), color in styles.items():
        selected = [
            row
            for row in rows
            if row["source"] == source
            and row["streak_type"] == kind
            and row["length_bin"] != f"{max_exact + 1}+"
        ]
        lengths = sorted({int(row["length_bin"]) for row in selected})
        means = [
            sum(
                float(row["observed_expected_ratio"])
                for row in selected
                if int(row["length_bin"]) == length
            )
            / sum(int(row["length_bin"]) == length for row in selected)
            for length in lengths
        ]
        ax.plot(
            lengths,
            means,
            marker="o",
            markersize=3,
            color=color,
            label=f"{source} {kind}",
        )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Streak length")
    ax.set_ylabel("Observed / geometric expected")
    ax.set_title("Observed-to-expected streak ratios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_outcome_autocorrelation(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for source in colors:
        selected = [
            row
            for row in rows
            if row["source"] == source
            and row["series"] == "net_per_initial_wager"
            and row.get("row_scope") == "aggregate"
        ]
        ax.errorbar(
            [row["lag"] for row in selected],
            [row["autocorrelation"] for row in selected],
            yerr=[
                _asymmetric_errors(
                    float(row["autocorrelation"]), row["student_t_95_ci"]
                )[0]
                for row in selected
            ],
            marker="o",
            color=colors[source],
            label=source,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Net-result autocorrelation")
    ax.set_title("Outcome autocorrelation by lag with seed uncertainty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_predictive_improvement(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    index = {
        (row["source"], row["seed"], row["target"], row["model"]): row
        for row in rows
        if row.get("seed") is not None
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    offsets = {"B": -0.12, "C": 0.12}
    for source, color in colors.items():
        seeds = sorted({key[1] for key in index if key[0] == source})
        for model in ("B", "C"):
            values = [
                index[(source, seed, "monetary", "A")]["mse"]
                - index[(source, seed, "monetary", model)]["mse"]
                for seed in seeds
            ]
            ax.scatter(
                [seed + offsets[model] for seed in seeds],
                values,
                color=color,
                marker="o" if model == "B" else "s",
                label=f"{source} model {model}",
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Independent seed")
    ax.set_ylabel("Held-out MSE A minus augmented model")
    ax.set_title("Held-out predictive MSE improvement by seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_conditional_streak_edge(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    order = [
        "loss_5_plus",
        "loss_4",
        "loss_3",
        "loss_2",
        "loss_1",
        "none",
        "win_1",
        "win_2",
        "win_3",
        "win_4",
        "win_5_plus",
    ]
    positions = {name: index for index, name in enumerate(order)}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for source, color in colors.items():
        selected = sorted(
            (row for row in rows if row["source"] == source),
            key=lambda row: positions[row["streak_group"]],
        )
        ax.plot(
            [positions[row["streak_group"]] for row in selected],
            [row["player_edge_per_initial_wager"] for row in selected],
            marker="o",
            color=color,
            label=source,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_ylabel("Next-round edge per initial wager")
    ax.set_title("Conditional next-round edge by live monetary streak")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_extreme_tail_plots(
    output_dir: Path,
    tails: Sequence[Mapping[str, Any]],
    slices: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    insurance: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Write the extreme-tail profitability and insurance plot set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    specs = (
        ("edge_by_nested_high_rich_tail.png", "tails"),
        ("edge_by_disjoint_score_slice.png", "slices"),
        ("one2six_minus_iid_tail_edge.png", "paired"),
        ("tail_frequency_vs_edge.png", "frequency"),
        ("tail_waiting_time.png", "waiting"),
        ("ace_up_insurance_probability_by_tail.png", "ace_up"),
        ("ten_up_ace_probability_by_tail.png", "ten_up"),
        ("insurance_ev_by_tail.png", "insurance_ev"),
    )
    for filename, kind in specs:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        if kind == "tails":
            _tail_edge_axes(ax, tails)
        elif kind == "slices":
            _slice_edge_axes(ax, slices)
        elif kind == "paired":
            rows = [row for row in paired if row["comparison"] == "neutral"]
            ax.bar([row["tail"] for row in rows], [row["mean"] for row in rows])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("One2Six minus IID contrast")
        elif kind == "frequency":
            rows = [
                row
                for row in tails
                if row["source"] == "one2six" and row["state"].startswith("high")
            ]
            ax.scatter(
                [row["frequency"] for row in rows], [row["mean_edge"] for row in rows]
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Eligible frequency")
            ax.set_ylabel("Player edge")
        elif kind == "waiting":
            rows = [row for row in opportunities if row["source"] == "one2six"]
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["tail"]].append(float(row["mean_wait"]))
            ax.bar(
                list(grouped),
                [sum(values) / len(values) for values in grouped.values()],
            )
            ax.set_ylabel("Mean rounds between eligible states")
        elif kind in {"ace_up", "ten_up"}:
            rows = [
                row
                for row in insurance
                if row["source"] == "one2six" and row["insurance_type"] == kind
            ]
            ax.bar(
                [row["state"] for row in rows],
                [row["mean_probability"] for row in rows],
            )
            ax.axhline(
                1 / 3 if kind == "ace_up" else 1 / 11, color="black", linestyle="--"
            )
            ax.set_ylabel("Insured-event probability")
        else:
            rows = [row for row in insurance if row["source"] == "one2six"]
            ax.bar(
                [f"{row['insurance_type']}:{row['state']}" for row in rows],
                [row["implied_ev_per_unit"] for row in rows],
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Insurance EV per unit")
        ax.set_title(filename.removesuffix(".png").replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)
        paths[filename.removesuffix(".png")] = filename
    return paths


def _tail_edge_axes(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for source, color in colors.items():
        selected = [
            row
            for row in rows
            if row["source"] == source and row["state"].startswith("high")
        ]
        ax.plot(
            [row["state"] for row in selected],
            [row["mean_edge"] for row in selected],
            marker="o",
            color=color,
            label=source,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Player edge per initial wager")
    ax.legend()


def _slice_edge_axes(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = {"physical_iid": "#4C78A8", "one2six": "#E45756"}
    for source, color in colors.items():
        selected = sorted(
            (row for row in rows if row["source"] == source),
            key=lambda row: row["slice_rank"],
        )
        ax.plot(
            [row["slice"] for row in selected],
            [row["player_edge"] for row in selected],
            marker="o",
            color=color,
            label=source,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Player edge per initial wager")
    ax.legend()
