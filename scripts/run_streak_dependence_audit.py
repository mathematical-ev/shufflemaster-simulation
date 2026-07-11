# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the monetary streak-shape and incremental predictive-value audit."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.streak_dependence_audit import (  # noqa: E402
    StreakDependenceAuditConfig,
    run_streak_dependence_audit,
)


def parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("values must be comma-separated integers.") from exc
    if not result:
        raise ValueError("at least one integer is required.")
    return result


def parse_args() -> StreakDependenceAuditConfig:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in range(72, 82))
    )
    parser.add_argument("--rounds-per-seed", type=int, default=100_000)
    parser.add_argument("--burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--max-exact-streak-length", type=int, default=20)
    parser.add_argument("--continuation-lengths", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--tail-thresholds", default="5,8,10,15,20")
    parser.add_argument("--autocorrelation-lags", default="1,2,3,5,10,20")
    parser.add_argument("--current-rack-weight", type=float, default=1.00)
    parser.add_argument("--returned-1-15-weight", type=float, default=0.75)
    parser.add_argument("--returned-16-50-weight", type=float, default=0.40)
    parser.add_argument("--returned-51-100-weight", type=float, default=0.20)
    parser.add_argument("--returned-over-100-weight", type=float, default=0.00)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/streak_dependence_audit_10x100k"),
    )
    args = parser.parse_args()
    return StreakDependenceAuditConfig(
        seeds=parse_ints(args.seeds),
        rounds_per_seed=args.rounds_per_seed,
        burn_in_rounds=args.burn_in_rounds,
        base_bet=args.base_bet,
        max_exact_streak_length=args.max_exact_streak_length,
        continuation_lengths=parse_ints(args.continuation_lengths),
        tail_thresholds=parse_ints(args.tail_thresholds),
        autocorrelation_lags=parse_ints(args.autocorrelation_lags),
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
    summary = run_streak_dependence_audit(config)
    print(f"Output directory: {config.output_dir}")
    print(f"Seeds: {config.seeds}")
    print(f"Measured rounds: {len(config.seeds) * config.rounds_per_seed * 2}")
    for source in ("physical_iid", "one2six"):
        metrics = summary["aggregate"][source]
        print(
            f"{source}: p(win resolved)={metrics['p_win_resolved']:.6f}, "
            f"p(loss resolved)={metrics['p_loss_resolved']:.6f}"
        )
    print(f"Conclusion: {summary['conclusion']}")
    print(f"Runtime: {perf_counter() - started:.3f} seconds")
    print("No betting or playing policy was selected.")


if __name__ == "__main__":
    main()
