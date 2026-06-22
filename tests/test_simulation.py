from shufflemaster_sim.simulation import SimulationConfig, run_star_blackjack_baseline


def test_star_blackjack_baseline_runs_requested_rounds() -> None:
    result = run_star_blackjack_baseline(SimulationConfig(rounds=10, seed=42))

    assert result.rounds_played == 10
    assert len(result.round_results) == 10
    assert result.initial_wagered == 100.0
    assert result.total_wagered > 0
    assert result.total_wagered == result.initial_wagered + result.action_wagered
    assert result.cumulative_profit == result.total_net_profit
    assert result.net_profit == result.total_net_profit


def test_star_blackjack_baseline_runs_with_finite_shoe_source() -> None:
    result = run_star_blackjack_baseline(
        SimulationConfig(rounds=25, seed=42, card_source="finite-shoe")
    )

    assert result.rounds_played == 25
    assert result.initial_wagered == 250.0
    assert result.total_wagered > 0


def test_star_blackjack_baseline_runs_with_manual_shoe_source() -> None:
    result = run_star_blackjack_baseline(
        SimulationConfig(rounds=25, seed=42, card_source="manual-shoe")
    )

    assert result.rounds_played == 25
    assert result.initial_wagered == 250.0
    assert result.total_wagered > 0
    assert result.box_results[0].max_win_streak >= 0
    assert result.box_results[0].max_loss_streak >= 0


def test_manual_shoe_simulation_reports_shuffle_count() -> None:
    result = run_star_blackjack_baseline(
        SimulationConfig(
            rounds=100,
            seed=42,
            card_source="manual-shoe",
            deck_count=1,
            cut_card_penetration=0.25,
        )
    )

    assert result.rounds_played == 100
    assert result.shuffle_count is not None
    assert result.shuffle_count > 0


def test_star_blackjack_baseline_runs_with_one2six_source() -> None:
    result = run_star_blackjack_baseline(
        SimulationConfig(rounds=10, seed=42, card_source="one2six")
    )

    assert result.rounds_played == 10
    assert result.output_buffer_size is not None
    assert result.ejection_count is not None
    assert result.fallback_ejection_count is not None


def test_star_blackjack_baseline_is_reproducible_with_same_seed() -> None:
    first = run_star_blackjack_baseline(SimulationConfig(rounds=100, seed=42))
    second = run_star_blackjack_baseline(SimulationConfig(rounds=100, seed=42))

    assert first.as_round_records() == second.as_round_records()
    assert first.as_box_records() == second.as_box_records()


def test_different_seeds_are_allowed_to_produce_different_results() -> None:
    first = run_star_blackjack_baseline(SimulationConfig(rounds=100, seed=1))
    second = run_star_blackjack_baseline(SimulationConfig(rounds=100, seed=2))

    assert first.rounds_played == second.rounds_played == 100
    assert first.total_wagered > 0
    assert second.total_wagered > 0
