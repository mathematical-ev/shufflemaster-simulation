from experiments.metrics import (
    geometric_probabilities,
    hilo_value,
    source_draw_metrics,
    streak_distributions,
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
