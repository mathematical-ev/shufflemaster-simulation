# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Sensitivity experiment for One2Six recurrence recycle batch size."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
from matplotlib import pyplot as plt

from experiments.one2six_recurrence import (
    DEFAULT_ONE2SIX_FIVE_SPADES,
    DEFAULT_ONE2SIX_TEN_SPADES,
    One2SixRecurrenceConfig,
    run_one2six_recurrence_experiment,
)

TAIL_LIMITS = (20, 50, 100, 250, 500, 1000)


@dataclass(frozen=True, slots=True)
class One2SixRecurrenceSensitivityConfig:
    """Configuration for recycle-batch-size sensitivity experiments."""

    draws: int = 1_000_000
    recycle_batch_sizes: tuple[int, ...] = (1, 5, 20, 52, 100)
    seed: int | None = 42
    output_dir: Path = Path(
        "experiments/outputs/one2six_recurrence_sensitivity_1m_seed42"
    )

    def __post_init__(self) -> None:
        if self.draws <= 0:
            raise ValueError("draws must be positive.")
        if not self.recycle_batch_sizes:
            raise ValueError("recycle_batch_sizes must not be empty.")
        if any(batch_size <= 0 for batch_size in self.recycle_batch_sizes):
            raise ValueError("all recycle_batch_sizes must be positive.")
        if len(set(self.recycle_batch_sizes)) != len(self.recycle_batch_sizes):
            raise ValueError("recycle_batch_sizes must be unique.")


def run_one2six_recurrence_sensitivity(
    config: One2SixRecurrenceSensitivityConfig,
) -> dict[str, Any]:
    """Run One2Six recurrence for each recycle batch size and summarize."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    physical_card_count: int | None = None
    comparator_probability: float | None = None

    for batch_size in config.recycle_batch_sizes:
        run_config = One2SixRecurrenceConfig(
            draws=config.draws,
            recycle_batch_size=batch_size,
            seed=config.seed,
            output_dir=config.output_dir / f"batch_{batch_size}",
        )
        metrics = run_one2six_recurrence_experiment(run_config)
        physical_card_count = int(metrics["physical_card_count"])
        comparator_probability = float(metrics["physical_iid_comparator_probability"])
        rows.append(summary_row(batch_size=batch_size, metrics=metrics))

    summary = {
        "experiment_name": "one2six_recurrence_recycle_batch_sensitivity",
        "draws": config.draws,
        "seed": config.seed,
        "recycle_batch_sizes": list(config.recycle_batch_sizes),
        "physical_card_count": physical_card_count,
        "physical_iid_comparator_probability": comparator_probability,
        "rows": rows,
        "assumptions_and_caveats": [
            "This is a source-level recurrence experiment, not blackjack EV.",
            "Physical IID is the null comparator.",
            "Deviations are recurrence-structure evidence, not "
            "exploitability evidence.",
            "recycle_batch_size is a modelling assumption for discard-return timing.",
        ],
    }
    summary["plot_paths"] = write_plots(config.output_dir, rows)
    write_json(config.output_dir / "summary.json", summary)
    write_csv(config.output_dir / "summary.csv", rows)
    write_markdown(config.output_dir / "summary.md", summary)
    return summary


def summary_row(*, batch_size: int, metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract flat summary fields from one One2Six recurrence metrics payload."""
    pooled = metrics["pooled_recurrence"]
    tails = pooled["tail_probabilities"]
    observed_tails = tails["observed"]
    iid_tails = tails["physical_iid_theoretical"]
    target_t = metrics["target_recurrence"][DEFAULT_ONE2SIX_TEN_SPADES]
    target_5 = metrics["target_recurrence"][DEFAULT_ONE2SIX_FIVE_SPADES]
    diagnostics = metrics["one2six_source_diagnostics"]
    group_summary = diagnostics["ejection_group_size_summary"]
    row: dict[str, Any] = {
        "recycle_batch_size": batch_size,
        "output_dir": metrics["output_dir"],
        "metrics_path": f"batch_{batch_size}/metrics.json",
        "pooled_mean_cards_between": pooled["mean_cards_between"],
        "pooled_median_cards_between": pooled["median_cards_between"],
        "pooled_quantile_75": pooled["quantiles"]["0.75"],
        "pooled_quantile_90": pooled["quantiles"]["0.90"],
        "pooled_quantile_95": pooled["quantiles"]["0.95"],
        "pooled_quantile_99": pooled["quantiles"]["0.99"],
        "target_t_mean_cards_between": target_t["mean_cards_between"],
        "target_5_mean_cards_between": target_5["mean_cards_between"],
        "target_t_appearances": target_t["appearances"],
        "target_5_appearances": target_5["appearances"],
        "ejection_count": diagnostics["ejection_count"],
        "fallback_ejection_count": diagnostics["fallback_ejection_count"],
        "fallback_ejection_rate": diagnostics["fallback_ejection_rate"],
        "mean_ejection_group_size": group_summary["mean"],
        "median_ejection_group_size": group_summary["median"],
        "final_output_buffer_size": diagnostics["final_output_buffer_size"],
        "accepted_discard_batch_count": diagnostics["accepted_discard_batch_count"],
        "invariant_check": diagnostics["invariant_check"],
        "pooled_chi_square_diagnostic": pooled["goodness_of_fit"][
            "chi_square_statistic"
        ],
    }
    for limit in TAIL_LIMITS:
        observed = float(observed_tails[str(limit)])
        theoretical = float(iid_tails[str(limit)])
        row[f"pooled_tail_le_{limit}"] = observed
        row[f"physical_iid_tail_le_{limit}"] = theoretical
        row[f"tail_minus_iid_le_{limit}"] = observed - theoretical
        row[f"tail_ratio_iid_le_{limit}"] = (
            observed / theoretical if theoretical else None
        )
    return row


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Write parent-level sensitivity plots and return relative paths."""
    plot_paths = {
        "pooled_tail_probabilities_by_batch_size": (
            "pooled_tail_probabilities_by_batch_size.png"
        ),
        "pooled_tail_probability_ratio_by_batch_size": (
            "pooled_tail_probability_ratio_by_batch_size.png"
        ),
        "pooled_mean_cards_between_by_batch_size": (
            "pooled_mean_cards_between_by_batch_size.png"
        ),
    }
    write_tail_probability_plot(
        output_dir / plot_paths["pooled_tail_probabilities_by_batch_size"],
        rows,
    )
    write_tail_ratio_plot(
        output_dir / plot_paths["pooled_tail_probability_ratio_by_batch_size"],
        rows,
    )
    write_mean_cards_between_plot(
        output_dir / plot_paths["pooled_mean_cards_between_by_batch_size"],
        rows,
    )
    return plot_paths


def write_tail_probability_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot pooled observed tail probabilities by recycle batch size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [row["recycle_batch_size"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    for limit in TAIL_LIMITS:
        y_values = [row[f"pooled_tail_le_{limit}"] for row in rows]
        ax.plot(x_values, y_values, marker="o", label=f"Observed <= {limit}")
        comparator = rows[0][f"physical_iid_tail_le_{limit}"]
        ax.axhline(comparator, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_title("Pooled Tail Probabilities By Recycle Batch Size")
    ax.set_xlabel("Recycle batch size")
    ax.set_ylabel("Observed tail probability")
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_tail_ratio_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot pooled observed/IID tail probability ratios by batch size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [row["recycle_batch_size"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    for limit in TAIL_LIMITS:
        y_values = [row[f"tail_ratio_iid_le_{limit}"] for row in rows]
        ax.plot(x_values, y_values, marker="o", label=f"<= {limit}")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Pooled Tail Probability Ratio To Physical IID")
    ax.set_xlabel("Recycle batch size")
    ax.set_ylabel("Observed / physical IID")
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_mean_cards_between_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot pooled mean cards-between by recycle batch size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [row["recycle_batch_size"] for row in rows],
        [row["pooled_mean_cards_between"] for row in rows],
        marker="o",
    )
    ax.set_title("Pooled Mean Cards Between By Recycle Batch Size")
    ax.set_xlabel("Recycle batch size")
    ax.set_ylabel("Mean cards between")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    columns = [
        "recycle_batch_size",
        "pooled_mean_cards_between",
        "pooled_median_cards_between",
        "pooled_tail_le_20",
        "pooled_tail_le_50",
        "pooled_tail_le_100",
        "pooled_tail_le_250",
        "pooled_tail_le_500",
        "pooled_tail_le_1000",
        "fallback_ejection_rate",
        "mean_ejection_group_size",
    ]
    lines = [
        "# One2Six Recurrence Sensitivity",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        row_values = [format_markdown_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(row_values) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is source-level only.",
            "- Physical IID is the null comparator.",
            "- Deviations are recurrence-structure evidence, not "
            "exploitability evidence.",
            "- `recycle_batch_size` is a modelling assumption for discard-return "
            "timing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_markdown_value(value: object) -> str:
    """Format values for the Markdown summary table."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
