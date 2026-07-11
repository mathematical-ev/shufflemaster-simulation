# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the one-box physical-IID versus One2Six game validation."""

import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.single_box_game_validation import (  # noqa: E402
    SingleBoxGameValidationConfig,
    run_single_box_game_validation,
)


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse a comma-separated seed list."""
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers.") from exc
    if not seeds:
        raise ValueError("at least one seed must be supplied.")
    return seeds


def parse_args() -> SingleBoxGameValidationConfig:
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1_000_000)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--include-pair-upcard-detail",
        action=BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/single_box_game_validation_1m"),
    )
    args = parser.parse_args()
    return SingleBoxGameValidationConfig(
        rounds=args.rounds,
        base_bet=args.base_bet,
        seeds=parse_seeds(args.seeds),
        output_dir=args.output_dir,
        include_pair_upcard_detail=args.include_pair_upcard_detail,
    )


def main() -> None:
    """Run the comparison and print its primary metrics."""
    config = parse_args()
    summary = run_single_box_game_validation(config)
    print(f"Output directory: {config.output_dir}")
    print(
        "source | net player | player blackjacks | doubles | pairs | splits | "
        "split given pair"
    )
    for source_name in ("physical_iid", "one2six"):
        metrics: dict[str, Any] = summary["aggregate"][source_name]
        print(
            f"{source_name} | "
            f"{metrics['net_player_result']:+.2f} | "
            f"{metrics['player_blackjacks']} "
            f"({metrics['player_blackjack_rate']:.4%}) | "
            f"{metrics['double_actions']} "
            f"({metrics['double_actions_per_round']:.4%}) | "
            f"{metrics['initial_pair_opportunities']} "
            f"({metrics['initial_pair_opportunity_rate']:.4%}) | "
            f"{metrics['split_actions']} "
            f"({metrics['split_actions_per_round']:.4%}) | "
            f"{metrics['split_rate_given_initial_pair']:.4%}"
        )


if __name__ == "__main__":
    main()
