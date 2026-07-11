# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from experiments.metrics import (
    MonetaryStreakTracker,
    classify_monetary_outcome,
    geometric_probabilities,
    hilo_value,
    signed_streak_histogram_data,
    source_draw_metrics,
    streak_distributions,
    streak_frequency_summary,
    validate_streak_reconciliation,
)
from experiments.runners import (
    IidBaselineExperimentConfig,
    run_iid_game_experiment,
)

from shufflemaster_sim.card_sources import IidRandomCardSource
from shufflemaster_sim.cards import RANKS


def test_source_experiment_metrics_for_iid_draws() -> None:
    source = IidRandomCardSource(seed=42)
    cards = [source.draw_card() for _ in range(1_000)]

    metrics = source_draw_metrics(
        cards,
        target_cards=("T:S", "5:S"),
        rank_targets=("T", "5"),
    )

    assert metrics["total_draws"] == 1_000
    assert sum(metrics["rank_counts"].values()) == 1_000
    assert sum(metrics["suit_counts"].values()) == 1_000
    assert metrics["hilo_values_seen"] == [-1, 0, 1]
    assert all(hilo_value(rank) in {-1, 0, 1} for rank in RANKS)
    assert all(gap > 0 for gap in metrics["target_card_recurrence"]["T:S"]["gaps"])
    assert all(gap > 0 for gap in metrics["rank_target_recurrence"]["T"]["gaps"])


def test_geometric_probabilities_are_sensible_over_plotted_support() -> None:
    probabilities = geometric_probabilities(250, 1.0 / 52.0)

    assert all(value >= 0 for value in probabilities.values())
    assert 0.99 < sum(probabilities.values()) < 1.0


def test_game_experiment_metrics_use_correct_denominators() -> None:
    metrics = run_iid_game_experiment(
        IidBaselineExperimentConfig(
            game_rounds=100,
            source_draws=100,
            run_source_experiment=False,
            seed=42,
        )
    )

    assert metrics["rounds"] == 100
    assert metrics["initial_hands"] == 100
    assert metrics["player_blackjack_rate_per_initial_hand"] == (
        metrics["player_blackjacks"] / metrics["initial_hands"]
    )
    assert metrics["edge_per_initial_wager"] == (
        metrics["net_profit"] / metrics["total_initial_wagered"]
    )
    assert metrics["edge_per_total_wager"] == (
        metrics["net_profit"] / metrics["total_wagered"]
    )


def test_game_metrics_from_result_accepts_simulation_result_shape() -> None:
    metrics = run_iid_game_experiment(
        IidBaselineExperimentConfig(
            game_rounds=10,
            source_draws=10,
            run_source_experiment=False,
            seed=7,
        )
    )

    assert "cumulative_profit_path" in metrics
    assert len(metrics["cumulative_profit_path"]) == 10


def test_streaks_win_win_push_win() -> None:
    assert streak_distributions([1, 1, 0, 1])["win_streaks"] == {3: 1}


def test_streaks_loss_loss_push_loss() -> None:
    assert streak_distributions([-1, -1, 0, -1])["loss_streaks"] == {3: 1}


def test_streaks_win_push_loss() -> None:
    streaks = streak_distributions([1, 0, -1])

    assert streaks["win_streaks"] == {1: 1}
    assert streaks["loss_streaks"] == {1: 1}


def test_streaks_push_push_win() -> None:
    assert streak_distributions([0, 0, 1])["win_streaks"] == {1: 1}


def test_streaks_mixed_sequence_with_pushes() -> None:
    streaks = streak_distributions([1, 0, 0, 1, -1, 0, -1])

    assert streaks["win_streaks"] == {2: 1}
    assert streaks["loss_streaks"] == {2: 1}


def test_signed_streak_distribution_uses_negative_losses() -> None:
    streaks = streak_distributions([1, 1, -1, 0, -1])

    assert streaks["signed_streaks"] == {-2: 1, 2: 1}


def test_all_pushes_create_no_streaks() -> None:
    streaks = streak_distributions([0, 0, 0])

    assert streaks["win_streaks"] == {}
    assert streaks["loss_streaks"] == {}


def test_leading_and_intervening_pushes_preserve_win_streak() -> None:
    assert streak_distributions([0, 0, 1, 0, 1])["win_streaks"] == {2: 1}


def test_pushes_do_not_break_open_win_or_loss_streaks() -> None:
    tracker = MonetaryStreakTracker()
    for outcome in (1, 0, 0, 1, -1, 0, 0, -1):
        tracker.observe(outcome)

    summary = tracker.summary()

    assert summary["win_streaks"]["frequency"] == {2: 1}
    assert summary["loss_streaks"]["frequency"] == {2: 1}


def test_open_streak_is_finalized_after_trailing_pushes() -> None:
    tracker = MonetaryStreakTracker()
    for outcome in (1, 0, 0):
        tracker.observe(outcome)

    assert tracker.summary()["win_streaks"]["frequency"] == {1: 1}


def test_monetary_outcome_classification() -> None:
    assert classify_monetary_outcome(10.0) == "win"
    assert classify_monetary_outcome(-10.0) == "loss"
    assert classify_monetary_outcome(0.0) == "push"


def test_streak_reconciliation_rejects_inconsistent_round_counts() -> None:
    tracker = MonetaryStreakTracker()
    tracker.observe(10.0)
    summary = tracker.summary()
    summary["rounds"] = 2

    with pytest.raises(RuntimeError, match="Round outcome counts"):
        validate_streak_reconciliation(summary)


def test_signed_histogram_data_uses_signs_and_overflow_without_mutation() -> None:
    wins = {1: 2, 4: 1, 21: 2, 30: 1}
    losses = {2: 3, 20: 1, 22: 2}
    original_wins = dict(wins)
    original_losses = dict(losses)

    histogram = signed_streak_histogram_data(wins, losses, display_limit=20)

    assert histogram["counts"][1] == 2
    assert histogram["counts"][4] == 1
    assert histogram["counts"][21] == 3
    assert histogram["counts"][-2] == 3
    assert histogram["counts"][-20] == 1
    assert histogram["counts"][-21] == 2
    assert 0 not in histogram["counts"]
    assert wins == original_wins
    assert losses == original_losses


def test_compact_streak_frequency_percentiles() -> None:
    summary = streak_frequency_summary({1: 1, 2: 2, 4: 1})

    assert summary["streak_count"] == 4
    assert summary["represented_rounds"] == 9
    assert summary["mean"] == 2.25
    assert summary["median"] == 2.0
    assert summary["p75"] == 2
    assert summary["p90"] == 4
    assert summary["p95"] == 4
    assert summary["p99"] == 4
    assert summary["maximum"] == 4
