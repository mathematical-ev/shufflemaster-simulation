# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

import subprocess
import sys
from pathlib import Path

from experiments.runners import (
    IidBaselineExperimentConfig,
    run_iid_baseline_experiment,
)


def test_iid_baseline_experiment_runner_creates_metrics_and_plots(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "iid_smoke"

    result = run_iid_baseline_experiment(
        IidBaselineExperimentConfig(
            source_draws=1_000,
            game_rounds=100,
            seed=42,
            output_dir=output_dir,
        )
    )

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "source_metrics.json").exists()
    assert (output_dir / "game_metrics.json").exists()
    assert result["source_metrics"]["total_draws"] == 1_000
    assert result["game_metrics"]["rounds"] == 100
    for plot_path in result["plot_paths"]:
        assert Path(plot_path).exists()


def test_iid_baseline_experiment_cli_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "iid_cli"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_iid_baseline_experiment.py",
            "--source-draws",
            "1000",
            "--game-rounds",
            "100",
            "--base-bet",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Output directory:" in completed.stdout
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "target_card_interarrival_T_S.png").exists()
    assert (output_dir / "target_card_interarrival_5_S.png").exists()
    assert (output_dir / "rank_interarrival_T.png").exists()
    assert (output_dir / "rank_interarrival_5.png").exists()
    assert (output_dir / "streak_histogram.png").exists()
    assert (output_dir / "streak_smoothed_density.png").exists()
    assert (output_dir / "outcome_percentages.png").exists()
    assert (output_dir / "cumulative_profit.png").exists()
