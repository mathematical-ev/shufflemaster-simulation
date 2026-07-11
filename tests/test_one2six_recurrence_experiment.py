# SPDX-License-Identifier: GPL-3.0-or-later

import subprocess
import sys
from pathlib import Path

from experiments.one2six_recurrence import (
    DEFAULT_ONE2SIX_FIVE_SPADES,
    DEFAULT_ONE2SIX_TEN_SPADES,
    One2SixRecurrenceConfig,
    calculate_one2six_recurrence_metrics,
    run_one2six_recurrence_experiment,
)


def test_one2six_recurrence_experiment_runs_small_count() -> None:
    metrics = calculate_one2six_recurrence_metrics(
        One2SixRecurrenceConfig(draws=1_000, recycle_batch_size=20, seed=42)
    )

    assert metrics["experiment_name"] == "one2six_physical_recurrence"
    assert metrics["card_source_type"] == "one2six"
    assert metrics["seed"] == 42
    assert metrics["draws"] == 1_000
    assert metrics["recycle_batch_size"] == 20
    assert "one2six_config" in metrics["config"]


def test_one2six_recurrence_target_and_pooled_summaries_exist() -> None:
    metrics = calculate_one2six_recurrence_metrics(
        One2SixRecurrenceConfig(draws=1_000, recycle_batch_size=20, seed=42)
    )

    assert DEFAULT_ONE2SIX_TEN_SPADES in metrics["target_recurrence"]
    assert DEFAULT_ONE2SIX_FIVE_SPADES in metrics["target_recurrence"]
    assert "pooled_recurrence" in metrics
    assert metrics["pooled_recurrence"]["return_observations"] > 0


def test_one2six_recurrence_tail_probabilities_match_keys() -> None:
    metrics = calculate_one2six_recurrence_metrics(
        One2SixRecurrenceConfig(draws=1_000, recycle_batch_size=20, seed=42)
    )
    summary = metrics["pooled_recurrence"]
    tails = summary["tail_probabilities"]

    assert set(tails["observed"]) == set(tails["physical_iid_theoretical"])
    assert all(0 <= value <= 1 for value in tails["observed"].values())
    assert all(0 <= value <= 1 for value in tails["physical_iid_theoretical"].values())


def test_one2six_recurrence_goodness_of_fit_bins_are_consistent() -> None:
    metrics = calculate_one2six_recurrence_metrics(
        One2SixRecurrenceConfig(draws=1_000, recycle_batch_size=20, seed=42)
    )
    summary = metrics["pooled_recurrence"]
    goodness = summary["goodness_of_fit"]

    assert goodness["bins"]
    assert (
        sum(bin_record["observed_count"] for bin_record in goodness["bins"])
        == summary["return_observations"]
    )
    assert all(
        bin_record["physical_iid_expected_count"] >= 0
        for bin_record in goodness["bins"]
    )
    assert goodness["chi_square_statistic"] >= 0


def test_one2six_source_diagnostics_are_present() -> None:
    metrics = calculate_one2six_recurrence_metrics(
        One2SixRecurrenceConfig(draws=1_000, recycle_batch_size=20, seed=42)
    )
    diagnostics = metrics["one2six_source_diagnostics"]

    assert diagnostics["ejection_count"] > 0
    assert diagnostics["fallback_ejection_count"] >= 0
    assert diagnostics["final_output_buffer_size"] >= 0
    assert "final_carousel_occupancy_summary" in diagnostics
    assert diagnostics["accepted_discard_batch_count"] > 0
    assert diagnostics["invariant_check"] == "passed"
    assert "ejection_group_size_distribution" in diagnostics


def test_one2six_recurrence_plot_paths_are_included() -> None:
    metrics = calculate_one2six_recurrence_metrics(
        One2SixRecurrenceConfig(draws=1_000, recycle_batch_size=20, seed=42)
    )

    plot_paths = metrics["plot_paths"]

    assert DEFAULT_ONE2SIX_TEN_SPADES in plot_paths["target_recurrence"]
    assert DEFAULT_ONE2SIX_FIVE_SPADES in plot_paths["target_recurrence"]
    assert (
        plot_paths["pooled_one2six_physical_recurrence"]
        == "pooled_one2six_physical_recurrence.png"
    )


def test_one2six_recurrence_runner_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "one2six"

    metrics = run_one2six_recurrence_experiment(
        One2SixRecurrenceConfig(
            draws=1_000,
            recycle_batch_size=20,
            seed=42,
            output_dir=output_dir,
        )
    )

    assert metrics["draws"] == 1_000
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "target_recurrence_one2six_deck_0_T_spades.png").exists()
    assert (output_dir / "target_recurrence_one2six_deck_0_5_spades.png").exists()
    assert (output_dir / "pooled_one2six_physical_recurrence.png").exists()


def test_one2six_recurrence_cli_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "one2six_cli"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_one2six_recurrence_experiment.py",
            "--draws",
            "1000",
            "--recycle-batch-size",
            "20",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Recycle batch size: 20" in completed.stdout
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "target_recurrence_one2six_deck_0_T_spades.png").exists()
    assert (output_dir / "target_recurrence_one2six_deck_0_5_spades.png").exists()
    assert (output_dir / "pooled_one2six_physical_recurrence.png").exists()
