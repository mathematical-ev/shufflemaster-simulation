# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

import subprocess
import sys
from pathlib import Path

from experiments.physical_iid import (
    DEFAULT_PHYSICAL_FIVE_SPADES,
    DEFAULT_PHYSICAL_TEN_SPADES,
    PhysicalIidRecurrenceConfig,
    calculate_physical_iid_recurrence_metrics,
    expected_cards_between_probabilities,
    recurrence_summary,
    run_physical_iid_recurrence_experiment,
)


def test_recurrence_summary_uses_draw_gap_minus_one() -> None:
    summary = recurrence_summary(
        total_draws=100,
        appearances=4,
        draw_gaps=[5, 10, 2],
        probability=1.0 / 312.0,
    )

    assert summary["cards_between_histogram"] == {1: 1, 4: 1, 9: 1}
    assert summary["min_cards_between"] == 1
    assert summary["max_cards_between"] == 9
    assert set(summary["tail_probabilities"]) == {"observed", "theoretical"}
    assert summary["goodness_of_fit"]["chi_square_statistic"] >= 0


def test_physical_iid_recurrence_metrics_populate_metadata_targets_and_pool() -> None:
    metrics = calculate_physical_iid_recurrence_metrics(
        PhysicalIidRecurrenceConfig(draws=5_000, deck_count=6, seed=42)
    )

    target = metrics["target_recurrence"][DEFAULT_PHYSICAL_TEN_SPADES]

    assert metrics["experiment_name"] == "six_deck_physical_iid_recurrence"
    assert metrics["card_source_type"] == "PhysicalIidCardSource"
    assert metrics["config"]["draws"] == 5_000
    assert metrics["config"]["deck_count"] == 6
    assert metrics["config"]["seed"] == 42
    assert metrics["seed"] == 42
    assert "geometric_model" in metrics
    assert metrics["physical_card_count"] == 312
    assert metrics["probability"] == 1.0 / 312.0
    assert target["total_draws"] == 5_000
    assert all(gap > 0 for gap in _histogram_to_gaps(target))
    assert metrics["pooled_recurrence"]["return_observations"] > 0


def test_expected_cards_between_probabilities_are_valid() -> None:
    probabilities = expected_cards_between_probabilities(
        physical_card_count=312,
        max_cards_between=1_000,
    )

    assert all(probability >= 0 for probability in probabilities.values())
    assert 0.95 < sum(probabilities.values()) < 1.0


def test_tail_probabilities_include_matching_observed_and_theoretical_keys() -> None:
    metrics = calculate_physical_iid_recurrence_metrics(
        PhysicalIidRecurrenceConfig(draws=5_000, deck_count=6, seed=42)
    )

    for summary in [
        metrics["target_recurrence"][DEFAULT_PHYSICAL_TEN_SPADES],
        metrics["target_recurrence"][DEFAULT_PHYSICAL_FIVE_SPADES],
        metrics["pooled_recurrence"],
    ]:
        tails = summary["tail_probabilities"]
        assert set(tails["observed"]) == set(tails["theoretical"])
        assert all(0 <= value <= 1 for value in tails["theoretical"].values())


def test_per_physical_card_count_diagnostics_are_consistent() -> None:
    metrics = calculate_physical_iid_recurrence_metrics(
        PhysicalIidRecurrenceConfig(draws=5_000, deck_count=6, seed=42)
    )

    diagnostics = metrics["per_physical_card_diagnostics"]
    appearance_distribution = diagnostics["appearance_counts"]["distribution"]
    return_distribution = diagnostics["return_counts"]["distribution"]
    appearance_total = sum(
        int(count) * frequency for count, frequency in appearance_distribution.items()
    )
    return_total = sum(
        int(count) * frequency for count, frequency in return_distribution.items()
    )

    assert diagnostics["appearance_counts"]["min"] >= 0
    assert diagnostics["return_counts"]["min"] >= 0
    assert appearance_total == metrics["draws"]
    assert return_total == metrics["pooled_recurrence"]["return_observations"]


def test_goodness_of_fit_diagnostics_are_consistent() -> None:
    metrics = calculate_physical_iid_recurrence_metrics(
        PhysicalIidRecurrenceConfig(draws=5_000, deck_count=6, seed=42)
    )

    for summary in [
        metrics["target_recurrence"][DEFAULT_PHYSICAL_TEN_SPADES],
        metrics["target_recurrence"][DEFAULT_PHYSICAL_FIVE_SPADES],
        metrics["pooled_recurrence"],
    ]:
        goodness = summary["goodness_of_fit"]
        observed_total = sum(
            bin_record["observed_count"] for bin_record in goodness["bins"]
        )
        assert observed_total == summary["return_observations"]
        assert all(bin_record["expected_count"] >= 0 for bin_record in goodness["bins"])
        assert goodness["chi_square_statistic"] >= 0


def test_physical_iid_recurrence_runner_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "physical_iid"

    metrics = run_physical_iid_recurrence_experiment(
        PhysicalIidRecurrenceConfig(
            draws=5_000,
            deck_count=6,
            seed=42,
            output_dir=output_dir,
        )
    )

    assert metrics["target_physical_ids"] == [
        DEFAULT_PHYSICAL_TEN_SPADES,
        DEFAULT_PHYSICAL_FIVE_SPADES,
    ]
    assert metrics["plot_paths"]["target_recurrence"][DEFAULT_PHYSICAL_TEN_SPADES] == (
        "target_recurrence_physical_iid_deck_0_T_spades.png"
    )
    assert (
        metrics["plot_paths"]["pooled_physical_recurrence"]
        == "pooled_physical_recurrence.png"
    )
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "target_recurrence_physical_iid_deck_0_T_spades.png").exists()
    assert (output_dir / "target_recurrence_physical_iid_deck_0_5_spades.png").exists()
    assert (output_dir / "pooled_physical_recurrence.png").exists()


def test_physical_iid_recurrence_cli_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "physical_iid_cli"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_physical_iid_recurrence_experiment.py",
            "--draws",
            "5000",
            "--deck-count",
            "6",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Physical cards: 312" in completed.stdout
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "target_recurrence_physical_iid_deck_0_T_spades.png").exists()
    assert (output_dir / "target_recurrence_physical_iid_deck_0_5_spades.png").exists()
    assert (output_dir / "pooled_physical_recurrence.png").exists()


def _histogram_to_gaps(summary: dict[str, object]) -> list[int]:
    histogram = summary["cards_between_histogram"]
    assert isinstance(histogram, dict)
    return [int(cards_between) + 1 for cards_between in histogram]
