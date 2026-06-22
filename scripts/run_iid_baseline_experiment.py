"""Run IID baseline source and game experiments."""

import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runners import (  # noqa: E402
    IidBaselineExperimentConfig,
    run_iid_baseline_experiment,
)


def parse_args() -> IidBaselineExperimentConfig:
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-draws", type=int, default=1_000_000)
    parser.add_argument("--game-rounds", type=int, default=1_000_000)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/iid_baseline"),
    )
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--game-only", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()
    if args.source_only and args.game_only:
        parser.error("--source-only and --game-only cannot be used together.")
    return IidBaselineExperimentConfig(
        source_draws=args.source_draws,
        game_rounds=args.game_rounds,
        base_bet=args.base_bet,
        seed=args.seed,
        output_dir=args.output_dir,
        run_source_experiment=not args.game_only,
        run_game_experiment=not args.source_only,
        save_raw=args.save_raw,
    )


def main() -> None:
    """Run the IID baseline experiment and print a compact summary."""
    config = parse_args()
    result = run_iid_baseline_experiment(config)
    source_metrics = result["source_metrics"]
    game_metrics = result["game_metrics"]

    print(f"Output directory: {config.output_dir}")
    if source_metrics is not None:
        print(f"Source draws: {source_metrics['total_draws']}")
    if game_metrics is not None:
        print(f"Game rounds: {game_metrics['rounds']}")
        print(
            "Player blackjack rate: "
            f"{game_metrics['player_blackjack_rate_per_initial_hand']:.4%}"
        )
        print(
            "Expected IID blackjack rate: "
            f"{game_metrics['expected_iid_player_blackjack_rate']:.4%}"
        )
        print(f"Edge per initial wager: {game_metrics['edge_per_initial_wager']:.4%}")
        print(f"Edge per total wager: {game_metrics['edge_per_total_wager']:.4%}")
    print("Generated plots:")
    for path in result["plot_paths"]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
