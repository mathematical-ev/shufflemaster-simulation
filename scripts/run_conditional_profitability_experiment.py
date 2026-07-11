# SPDX-License-Identifier: GPL-3.0-or-later

"""Run held-out conditional profitability for the frozen observable score."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.conditional_profitability import (  # noqa: E402
    ConditionalProfitabilityConfig,
    run_conditional_profitability_experiment,
)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse a nonempty comma-separated integer tuple."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("values must be comma-separated integers.") from exc
    if not values:
        raise ValueError("at least one integer is required.")
    return values


def parse_float_tuple(value: str) -> tuple[float, ...]:
    """Parse a nonempty comma-separated float tuple."""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("values must be comma-separated numbers.") from exc
    if not values:
        raise ValueError("at least one number is required.")
    return values


def parse_args() -> ConditionalProfitabilityConfig:
    """Parse the frozen experiment configuration."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--development-seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--validation-seeds", default="52,53,54,55,56,57,58,59,60,61")
    parser.add_argument("--development-rounds-per-seed", type=int, default=20_000)
    parser.add_argument("--validation-rounds-per-seed", type=int, default=100_000)
    parser.add_argument("--burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--score-quantiles", default="0.10,0.30,0.70,0.90")
    parser.add_argument("--current-rack-weight", type=float, default=1.00)
    parser.add_argument("--returned-1-15-weight", type=float, default=0.75)
    parser.add_argument("--returned-16-50-weight", type=float, default=0.40)
    parser.add_argument("--returned-51-100-weight", type=float, default=0.20)
    parser.add_argument("--returned-over-100-weight", type=float, default=0.00)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/conditional_profitability_validation"),
    )
    args = parser.parse_args()
    return ConditionalProfitabilityConfig(
        development_seeds=parse_int_tuple(args.development_seeds),
        validation_seeds=parse_int_tuple(args.validation_seeds),
        development_rounds_per_seed=args.development_rounds_per_seed,
        validation_rounds_per_seed=args.validation_rounds_per_seed,
        burn_in_rounds=args.burn_in_rounds,
        base_bet=args.base_bet,
        score_quantiles=parse_float_tuple(args.score_quantiles),
        current_rack_weight=args.current_rack_weight,
        returned_1_15_weight=args.returned_1_15_weight,
        returned_16_50_weight=args.returned_16_50_weight,
        returned_51_100_weight=args.returned_51_100_weight,
        returned_over_100_weight=args.returned_over_100_weight,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run the experiment and print its monetary headline."""
    config = parse_args()
    started = perf_counter()
    summary = run_conditional_profitability_experiment(config)
    print(f"Output directory: {config.output_dir}")
    print(f"Development seeds: {config.development_seeds}")
    print(f"Validation seeds: {config.validation_seeds}")
    print(f"Frozen cutpoints: {summary['score_band_cutpoints']['cutpoints']}")
    print("Continuous monetary response:")
    for row in summary["continuous_monetary_slopes"]:
        print(
            f"  {row['source']}: slope={row['mean_monetary_slope']:.6f} "
            f"CI={row['student_t_95_ci']}"
        )
    paired = summary["paired_source_monetary_slopes"][0]
    print(
        "Paired One2Six-minus-IID slope: "
        f"{paired['mean_difference']:.6f} CI={paired['student_t_95_ci']}"
    )
    candidates = [
        row["score_band"]
        for row in summary["candidate_positive_ev_states"]
        if row["candidate_positive_ev_state"]
    ]
    print(f"Candidate positive-EV states: {candidates or 'none under fixed criteria'}")
    print(f"Runtime: {perf_counter() - started:.3f} seconds")
    print("No strategy, wager, threshold, or box-count policy was selected.")


if __name__ == "__main__":
    main()
