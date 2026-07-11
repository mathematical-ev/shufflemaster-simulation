# SPDX-License-Identifier: GPL-3.0-or-later

"""Run score-conditioned paired player-action values."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.score_conditioned_action_values import (  # noqa: E402
    ScoreConditionedActionValueConfig,
    run_score_conditioned_action_values,
)


def parse_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers.") from exc
    if not values:
        raise ValueError("at least one seed is required.")
    return values


def parse_quantiles(value: str) -> tuple[float, float]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("quantiles must be comma-separated numbers.") from exc
    if len(values) != 2:
        raise ValueError("exactly two composition quantiles are required.")
    return values[0], values[1]


def parse_args() -> ScoreConditionedActionValueConfig:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--development-seeds", default="62,63,64,65,66")
    parser.add_argument("--validation-seeds", default="67,68,69,70,71")
    parser.add_argument("--decision-states-per-seed", type=int, default=10_000)
    parser.add_argument("--burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--composition-quantiles", default="0.30,0.70")
    parser.add_argument("--current-weight", type=float, default=1.00)
    parser.add_argument("--returned-1-15-weight", type=float, default=0.75)
    parser.add_argument("--returned-16-50-weight", type=float, default=0.40)
    parser.add_argument("--returned-51-100-weight", type=float, default=0.20)
    parser.add_argument("--returned-over-100-weight", type=float, default=0.00)
    parser.add_argument("--minimum-total-state-count", type=int, default=500)
    parser.add_argument("--minimum-per-seed-state-count", type=int, default=50)
    parser.add_argument("--minimum-seed-sign-count", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/score_conditioned_action_values"),
    )
    args = parser.parse_args()
    return ScoreConditionedActionValueConfig(
        development_seeds=parse_ints(args.development_seeds),
        validation_seeds=parse_ints(args.validation_seeds),
        decision_states_per_seed=args.decision_states_per_seed,
        burn_in_rounds=args.burn_in_rounds,
        base_bet=args.base_bet,
        composition_quantiles=parse_quantiles(args.composition_quantiles),
        current_weight=args.current_weight,
        returned_1_15_weight=args.returned_1_15_weight,
        returned_16_50_weight=args.returned_16_50_weight,
        returned_51_100_weight=args.returned_51_100_weight,
        returned_over_100_weight=args.returned_over_100_weight,
        minimum_total_state_count=args.minimum_total_state_count,
        minimum_per_seed_state_count=args.minimum_per_seed_state_count,
        minimum_seed_sign_count=args.minimum_seed_sign_count,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    started = perf_counter()
    summary = run_score_conditioned_action_values(config)
    print(f"Output directory: {config.output_dir}")
    print(f"Development seeds: {config.development_seeds}")
    print(f"Validation seeds: {config.validation_seeds}")
    print(f"Composition cutpoints: {summary['composition_cutpoints']['cutpoints']}")
    print(f"Development candidates: {summary['development_candidate_count']}")
    print(f"Validated candidates: {summary['validated_candidate_count']}")
    print(f"Generic corrections: {len(summary['generic_strategy_corrections'])}")
    print(f"One2Six-specific deviations: {len(summary['one2six_specific_deviations'])}")
    print(f"Loss reductions: {len(summary['loss_reduction_deviations'])}")
    print(f"Edge creations: {len(summary['edge_creation_deviations'])}")
    print(f"Runtime: {perf_counter() - started:.3f} seconds")
    print("No revised strategy, betting policy, or box-count policy was deployed.")


if __name__ == "__main__":
    main()
