# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

from pathlib import Path

from experiments.plots import (
    plot_cumulative_profit,
    plot_interarrival_histogram,
    plot_outcome_percentages,
    plot_signed_streak_histogram,
    plot_signed_streak_smoothed_density,
)


def assert_png_created(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_functions_create_png_files(tmp_path: Path) -> None:
    recurrence = {
        "cards_between_histogram": {
            0: 2,
            1: 3,
            2: 1,
        }
    }
    streaks = {
        "win_streaks": {1: 2, 2: 1},
        "loss_streaks": {1: 1},
        "signed_streaks": {-1: 1, 1: 2, 2: 1},
    }
    theoretical = {
        "win": {1: 0.5, 2: 0.25},
        "loss": {1: 0.5},
    }
    game_metrics = {
        "win_rate_per_initial_round": 0.44,
        "loss_rate_per_initial_round": 0.48,
        "push_rate_per_initial_round": 0.08,
        "player_blackjack_rate_per_initial_hand": 0.047,
        "split_rate_per_initial_hand": 0.03,
        "double_rate_per_initial_hand": 0.09,
        "bust_rate_per_resolved_hand": 0.16,
    }

    paths = [
        plot_interarrival_histogram(
            recurrence,
            probability=1.0 / 52.0,
            title="Target",
            output_path=tmp_path / "target.png",
        ),
        plot_signed_streak_histogram(
            streaks,
            theoretical=theoretical,
            output_path=tmp_path / "streak_histogram.png",
        ),
        plot_signed_streak_smoothed_density(
            streaks,
            output_path=tmp_path / "streak_density.png",
        ),
        plot_outcome_percentages(
            game_metrics,
            output_path=tmp_path / "outcome_percentages.png",
        ),
        plot_cumulative_profit(
            [10.0, 0.0, 15.0],
            output_path=tmp_path / "cumulative_profit.png",
        ),
    ]

    for path in paths:
        assert_png_created(path)
