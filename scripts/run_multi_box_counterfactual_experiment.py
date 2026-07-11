# SPDX-License-Identifier: GPL-3.0-or-later

"""Run common-state counterfactual next-round box-count branches."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.multi_box_counterfactual import (  # noqa: E402
    STATE_BUCKETS,
    MultiBoxCounterfactualConfig,
    run_multi_box_counterfactual_experiment,
)


def parse_integer_list(value: str, *, field_name: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated integer tuple."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be comma-separated integers.") from exc
    if not values:
        raise ValueError(f"at least one {field_name} value must be supplied.")
    return values


def parse_args() -> MultiBoxCounterfactualConfig:
    """Parse command-line configuration."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--states-per-seed", type=int, default=2_000)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--sample-interval-rounds", type=int, default=5)
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--box-counts", default="1,2,3,4,5,6,7")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/multi_box_counterfactual_5x2000"),
    )
    args = parser.parse_args()
    return MultiBoxCounterfactualConfig(
        states_per_seed=args.states_per_seed,
        seeds=parse_integer_list(args.seeds, field_name="seeds"),
        burn_in_rounds=args.burn_in_rounds,
        sample_interval_rounds=args.sample_interval_rounds,
        base_bet=args.base_bet,
        box_counts=parse_integer_list(args.box_counts, field_name="box counts"),
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run the experiment and print descriptive action-value summaries."""
    config = parse_args()
    started = perf_counter()
    summary = run_multi_box_counterfactual_experiment(config)
    runtime = perf_counter() - started
    print(f"Output directory: {config.output_dir}")
    print(
        f"States: {2 * len(config.seeds) * config.states_per_seed:,}; "
        f"branches: "
        f"{2 * len(config.seeds) * config.states_per_seed * len(config.box_counts):,}"
    )
    frequency = summary["state_frequency"]
    actions = summary["state_action_values"]
    marginals = [
        row
        for row in summary["marginal_box_values"]
        if row["seed_or_aggregate"] == "aggregate"
    ]
    for source_name in ("physical_iid", "one2six"):
        counts = {
            row["state_bucket"]: row["state_count"]
            for row in frequency
            if row["source"] == source_name and row["seed_or_aggregate"] == "aggregate"
        }
        print(f"\n{source_name} state counts: {counts}")
        for bucket in STATE_BUCKETS:
            one = _action_row(actions, source_name, bucket, 1)
            seven = _action_row(actions, source_name, bucket, 7)
            marginal_values = {
                row["added_box_number"]: row["marginal_return_per_added_box"]
                for row in marginals
                if row["source"] == source_name and row["state_bucket"] == bucket
            }
            print(
                f"  {bucket}: one-box={_percent(one)}; "
                f"seven-box={_percent(seven)}; marginals={marginal_values}"
            )
    print(f"\nRuntime: {runtime:.3f} seconds")
    print("No box-count policy was selected.")


def _action_row(
    rows: list[dict[str, Any]],
    source: str,
    bucket: str,
    box_count: int,
) -> float | None:
    matches = [
        row["edge_per_initial_wager"]
        for row in rows
        if row["source"] == source
        and row["state_bucket"] == bucket
        and row["box_count"] == box_count
    ]
    return matches[0] if matches else None


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3%}"


if __name__ == "__main__":
    main()
