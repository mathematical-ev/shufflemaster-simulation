# SPDX-License-Identifier: GPL-3.0-or-later

"""Run held-out validation of the frozen observable fading signal."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fading_exclusion_validation import (  # noqa: E402
    FROZEN_WEIGHTS,
    FadingExclusionValidationConfig,
    run_fading_exclusion_validation,
)


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse comma-separated held-out seeds."""
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers.") from exc
    if not seeds:
        raise ValueError("at least one seed must be supplied.")
    return seeds


def parse_args() -> FadingExclusionValidationConfig:
    """Parse held-out experiment configuration."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="47,48,49,50,51")
    parser.add_argument("--rounds-per-seed", type=int, default=50_000)
    parser.add_argument("--burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--probe-states-per-seed", type=int, default=3_000)
    parser.add_argument("--probe-cards", type=int, default=15)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--current-rack-weight", type=float, default=1.00)
    parser.add_argument("--returned-1-15-weight", type=float, default=0.75)
    parser.add_argument("--returned-16-50-weight", type=float, default=0.40)
    parser.add_argument("--returned-51-100-weight", type=float, default=0.20)
    parser.add_argument("--returned-over-100-weight", type=float, default=0.00)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/fading_exclusion_validation_heldout"),
    )
    args = parser.parse_args()
    weights = {
        "current_rack": args.current_rack_weight,
        "returned_1_15": args.returned_1_15_weight,
        "returned_16_50": args.returned_16_50_weight,
        "returned_51_100": args.returned_51_100_weight,
        "returned_over_100": args.returned_over_100_weight,
    }
    return FadingExclusionValidationConfig(
        seeds=parse_seeds(args.seeds),
        rounds_per_seed=args.rounds_per_seed,
        burn_in_rounds=args.burn_in_rounds,
        probe_states_per_seed=args.probe_states_per_seed,
        probe_cards=args.probe_cards,
        base_bet=args.base_bet,
        current_rack_weight=args.current_rack_weight,
        returned_1_15_weight=args.returned_1_15_weight,
        returned_16_50_weight=args.returned_16_50_weight,
        returned_51_100_weight=args.returned_51_100_weight,
        returned_over_100_weight=args.returned_over_100_weight,
        allow_weight_override=weights != FROZEN_WEIGHTS,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run validation and print primary held-out results."""
    config = parse_args()
    started = perf_counter()
    summary = run_fading_exclusion_validation(config)
    print(f"Output directory: {config.output_dir}")
    print(f"Held-out seeds: {config.seeds}; frozen weights: {config.weights}")
    print("Primary next-15 slopes:")
    for row in summary["next15_primary_validation"]:
        print(
            f"  {row['source']} {row['outcome']}: "
            f"{row['mean_seed_slope']:.4f} CI={row['student_t_95_ci']}"
        )
    print("Paired One2Six-minus-IID differences:")
    for row in summary["paired_source_slope_differences"]:
        print(
            f"  {row['outcome']}: {row['mean_paired_slope_difference']:.4f} "
            f"CI={row['student_t_95_ci']}"
        )
    print(f"Runtime: {perf_counter() - started:.3f} seconds")
    print("No weights were fitted and no betting or box-count policy was selected.")


if __name__ == "__main__":
    main()
