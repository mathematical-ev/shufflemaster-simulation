# SPDX-License-Identifier: GPL-3.0-or-later

"""Run extreme-tail profitability and insurance feasibility."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.extreme_tail_profitability import (  # noqa: E402
    ExtremeTailProfitabilityConfig,
    run_extreme_tail_profitability,
)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> ExtremeTailProfitabilityConfig:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--development-seeds", default="82,83,84,85,86")
    parser.add_argument("--validation-seeds", default="87,88,89,90,91,92,93,94,95,96")
    parser.add_argument("--development-rounds-per-seed", type=int, default=20_000)
    parser.add_argument("--validation-rounds-per-seed", type=int, default=100_000)
    parser.add_argument("--burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--high-rich-quantiles", default="0.20,0.10,0.05,0.025,0.01")
    parser.add_argument("--neutral-quantiles", default="0.40,0.60")
    parser.add_argument("--current-rack-weight", type=float, default=1.0)
    parser.add_argument("--returned-1-15-weight", type=float, default=0.75)
    parser.add_argument("--returned-16-50-weight", type=float, default=0.40)
    parser.add_argument("--returned-51-100-weight", type=float, default=0.20)
    parser.add_argument("--returned-over-100-weight", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/extreme_tail_profitability"),
    )
    args = parser.parse_args()
    neutral = parse_floats(args.neutral_quantiles)
    if len(neutral) != 2:
        raise ValueError("neutral quantiles require exactly two values.")
    return ExtremeTailProfitabilityConfig(
        development_seeds=parse_ints(args.development_seeds),
        validation_seeds=parse_ints(args.validation_seeds),
        development_rounds_per_seed=args.development_rounds_per_seed,
        validation_rounds_per_seed=args.validation_rounds_per_seed,
        burn_in_rounds=args.burn_in_rounds,
        base_bet=args.base_bet,
        high_rich_quantiles=parse_floats(args.high_rich_quantiles),
        neutral_quantiles=(neutral[0], neutral[1]),
        current_rack_weight=args.current_rack_weight,
        returned_1_15_weight=args.returned_1_15_weight,
        returned_16_50_weight=args.returned_16_50_weight,
        returned_51_100_weight=args.returned_51_100_weight,
        returned_over_100_weight=args.returned_over_100_weight,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    started = perf_counter()
    summary = run_extreme_tail_profitability(config)
    print(f"Output directory: {config.output_dir}")
    print(f"Cutpoints: {summary['cutpoints']['values']}")
    validated = sum(
        row["validated_positive_ev_state"] for row in summary["positive_ev_state_gate"]
    )
    print(f"Validated positive-EV tails: {validated}")
    print(f"Runtime: {perf_counter() - started:.3f} seconds")
    print("No betting, playing, or box-count policy was selected.")


if __name__ == "__main__":
    main()
