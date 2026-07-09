import csv
import subprocess
import sys
from pathlib import Path

from experiments.one2six_recurrence_sensitivity import (
    One2SixRecurrenceSensitivityConfig,
    run_one2six_recurrence_sensitivity,
)


def test_one2six_recurrence_sensitivity_creates_batch_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "sensitivity"

    summary = run_one2six_recurrence_sensitivity(
        One2SixRecurrenceSensitivityConfig(
            draws=1_000,
            recycle_batch_sizes=(1, 5, 20),
            seed=42,
            output_dir=output_dir,
        )
    )

    assert summary["recycle_batch_sizes"] == [1, 5, 20]
    for batch_size in (1, 5, 20):
        assert (output_dir / f"batch_{batch_size}" / "metrics.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()


def test_one2six_recurrence_sensitivity_summary_rows_are_complete(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "sensitivity"

    summary = run_one2six_recurrence_sensitivity(
        One2SixRecurrenceSensitivityConfig(
            draws=1_000,
            recycle_batch_sizes=(1, 5, 20),
            seed=42,
            output_dir=output_dir,
        )
    )

    assert len(summary["rows"]) == 3
    for row in summary["rows"]:
        assert "pooled_tail_le_20" in row
        assert "tail_minus_iid_le_20" in row
        assert "tail_ratio_iid_le_20" in row
        assert "fallback_ejection_count" in row
        assert "fallback_ejection_rate" in row
        assert row["invariant_check"] == "passed"
        assert row["accepted_discard_batch_count"] > 0


def test_one2six_recurrence_sensitivity_csv_has_one_row_per_batch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "sensitivity"

    run_one2six_recurrence_sensitivity(
        One2SixRecurrenceSensitivityConfig(
            draws=1_000,
            recycle_batch_sizes=(1, 5, 20),
            seed=42,
            output_dir=output_dir,
        )
    )

    with (output_dir / "summary.csv").open(encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    assert [row["recycle_batch_size"] for row in rows] == ["1", "5", "20"]


def test_one2six_recurrence_sensitivity_plot_paths_are_included(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "sensitivity"

    summary = run_one2six_recurrence_sensitivity(
        One2SixRecurrenceSensitivityConfig(
            draws=1_000,
            recycle_batch_sizes=(1, 5, 20),
            seed=42,
            output_dir=output_dir,
        )
    )

    assert "pooled_tail_probabilities_by_batch_size" in summary["plot_paths"]
    assert "pooled_tail_probability_ratio_by_batch_size" in summary["plot_paths"]
    assert "pooled_mean_cards_between_by_batch_size" in summary["plot_paths"]
    for path in summary["plot_paths"].values():
        assert (output_dir / path).exists()


def test_one2six_recurrence_sensitivity_cli_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "sensitivity_cli"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_one2six_recurrence_sensitivity.py",
            "--draws",
            "1000",
            "--recycle-batch-sizes",
            "1,5,20",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Recycle batch sizes: 1,5,20" in completed.stdout
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
