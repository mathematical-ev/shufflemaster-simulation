# SPDX-License-Identifier: GPL-3.0-or-later

"""Run observable current-rack and returned-batch response analysis."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.observable_card_response import (  # noqa: E402
    ObservableCardResponseConfig,
    run_observable_card_response_experiment,
)


def parse_integer_list(value: str, *, field_name: str) -> tuple[int, ...]:
    """Parse comma-separated integers."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be comma-separated integers.") from exc
    if not values:
        raise ValueError(f"at least one {field_name} value must be supplied.")
    return values


def parse_lag_bands(value: str) -> tuple[tuple[int, int], ...]:
    """Parse comma-separated inclusive ranges such as 1-15,16-50."""
    bands: list[tuple[int, int]] = []
    try:
        for item in value.split(","):
            start, end = item.strip().split("-", maxsplit=1)
            bands.append((int(start), int(end)))
    except ValueError as exc:
        raise ValueError(
            "lag bands must use comma-separated start-end ranges."
        ) from exc
    if not bands:
        raise ValueError("at least one lag band must be supplied.")
    return tuple(bands)


def parse_args() -> ObservableCardResponseConfig:
    """Parse command-line experiment configuration."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--base-bet", type=float, default=10.0)
    parser.add_argument("--current-rack-states-per-seed", type=int, default=3_000)
    parser.add_argument("--current-rack-burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--current-rack-sample-interval-rounds", type=int, default=5)
    parser.add_argument("--current-rack-probe-cards", type=int, default=15)
    parser.add_argument("--lag-rounds-per-seed", type=int, default=50_000)
    parser.add_argument("--lag-burn-in-rounds", type=int, default=1_000)
    parser.add_argument("--lag-horizon-cards", type=int, default=1_000)
    parser.add_argument(
        "--lag-bands",
        default="1-15,16-50,51-100,101-250,251-500,501-1000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/observable_card_response_5seed"),
    )
    args = parser.parse_args()
    return ObservableCardResponseConfig(
        seeds=parse_integer_list(args.seeds, field_name="seeds"),
        base_bet=args.base_bet,
        current_rack_states_per_seed=args.current_rack_states_per_seed,
        current_rack_burn_in_rounds=args.current_rack_burn_in_rounds,
        current_rack_sample_interval_rounds=(args.current_rack_sample_interval_rounds),
        current_rack_probe_cards=args.current_rack_probe_cards,
        lag_rounds_per_seed=args.lag_rounds_per_seed,
        lag_burn_in_rounds=args.lag_burn_in_rounds,
        lag_horizon_cards=args.lag_horizon_cards,
        lag_bands=parse_lag_bands(args.lag_bands),
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run both response experiments and print their headline slopes."""
    config = parse_args()
    started = perf_counter()
    summary = run_observable_card_response_experiment(config)
    print(f"Output directory: {config.output_dir}")
    print("Current-rack next-15 mean seed slopes:")
    for row in summary["current_rack_primary_response"]:
        if row["outcome"] in {"hi_lo", "low", "ten_value", "ace"}:
            print(
                f"  {row['source']} {row['outcome']}: "
                f"{_format(row['mean_seed_slope'])} "
                f"CI={row['student_t_95_ci']}"
            )
    print("One2Six returned-batch lag-band slopes:")
    for row in summary["returned_batch_lag_response"]:
        if row["source"] == "one2six" and row["outcome"] in {
            "hi_lo",
            "low",
            "ten_value",
            "ace",
        }:
            print(f"  {row['lag']} {row['outcome']}: {_format(row['mean_seed_slope'])}")
    print(f"Runtime: {perf_counter() - started:.3f} seconds")
    print("No betting, box-count, or bankroll policy was selected.")


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
