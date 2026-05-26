"""Run the Star Blackjack baseline simulation."""

from argparse import ArgumentParser

from shufflemaster_sim.simulation import (
    SimulationConfig,
    run_star_blackjack_baseline,
)


def parse_box_bets(raw_box_bets: str | None) -> dict[int, float] | None:
    """Parse box bets from strings like '1:10,2:25'."""
    if raw_box_bets is None:
        return None

    box_bets: dict[int, float] = {}
    for item in raw_box_bets.split(","):
        box_id_text, bet_text = item.split(":", maxsplit=1)
        box_bets[int(box_id_text)] = float(bet_text)
    return box_bets


def parse_args() -> SimulationConfig:
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10_000)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--box-bets", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--box-count", type=int, default=1)
    parser.add_argument(
        "--card-source",
        choices=["iid", "finite-shoe", "manual-shoe"],
        default="iid",
    )
    parser.add_argument("--deck-count", type=int, default=None)
    parser.add_argument("--cut-card-penetration", type=float, default=0.75)
    args = parser.parse_args()
    return SimulationConfig(
        rounds=args.rounds,
        base_bet=args.base_bet,
        box_count=args.box_count,
        box_bets=parse_box_bets(args.box_bets),
        seed=args.seed,
        card_source=args.card_source,
        deck_count=args.deck_count,
        cut_card_penetration=args.cut_card_penetration,
    )


def main() -> None:
    """Run the baseline and print a compact summary."""
    config = parse_args()
    result = run_star_blackjack_baseline(config)
    box = result.box_results[0]
    player_edge_initial = result.edge_per_initial_wager
    player_edge_total = result.edge_per_total_wager

    print(f"Rounds: {result.rounds_played}")
    print(f"Base bet: ${config.base_bet:,.2f}")
    print(f"Card source: {config.card_source}")
    print(f"Deck count: {config.effective_deck_count}")
    if config.card_source == "manual-shoe":
        print(f"Cut-card penetration: {config.cut_card_penetration:.2%}")
    if result.shuffle_count is not None:
        print(f"Shuffle count: {result.shuffle_count}")
    print(f"Initial wagered: ${result.initial_wagered:,.2f}")
    print(f"Action wagered: ${result.action_wagered:,.2f}")
    print(f"Total wagered: ${result.total_wagered:,.2f}")
    print(f"Net profit: ${result.net_profit:,.2f}")
    print(f"Average profit per round: ${result.average_profit_per_round:,.4f}")
    print(f"Player edge per initial wager: {player_edge_initial:.4%}")
    print(f"House edge per initial wager: {-player_edge_initial:.4%}")
    print(f"Player edge per total wager: {player_edge_total:.4%}")
    print(f"House edge per total wager: {-player_edge_total:.4%}")
    print(f"Wins: {box.wins}")
    print(f"Losses: {box.losses}")
    print(f"Pushes: {box.pushes}")
    print(f"Blackjacks: {box.blackjacks}")
    print(f"Doubles: {box.doubles}")
    print(f"Splits: {box.splits}")
    print(f"Busts: {box.busts}")
    print(f"Box {box.box_id} max win streak: {box.max_win_streak}")
    print(f"Box {box.box_id} max loss streak: {box.max_loss_streak}")


if __name__ == "__main__":
    main()
