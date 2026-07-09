# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Run One2Six recurrence sensitivity over recycle batch size."""

import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.one2six_recurrence_sensitivity import (  # noqa: E402
    One2SixRecurrenceSensitivityConfig,
    run_one2six_recurrence_sensitivity,
)


def parse_batch_sizes(raw_value: str) -> tuple[int, ...]:
    """Parse comma-separated recycle batch sizes."""
    batch_sizes = tuple(
        int(item.strip()) for item in raw_value.split(",") if item.strip()
    )
    if not batch_sizes:
        raise ValueError("At least one recycle batch size is required.")
    return batch_sizes


def parse_args() -> One2SixRecurrenceSensitivityConfig:
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1_000_000)
    parser.add_argument("--recycle-batch-sizes", type=str, default="1,5,20,52,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/one2six_recurrence_sensitivity_1m_seed42"),
    )
    args = parser.parse_args()
    return One2SixRecurrenceSensitivityConfig(
        draws=args.draws,
        recycle_batch_sizes=parse_batch_sizes(args.recycle_batch_sizes),
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run the sensitivity experiment and print a compact summary."""
    config = parse_args()
    summary = run_one2six_recurrence_sensitivity(config)

    print(f"Output directory: {config.output_dir}")
    print(f"Draws per batch size: {config.draws}")
    batch_sizes = ",".join(str(size) for size in config.recycle_batch_sizes)
    print(f"Recycle batch sizes: {batch_sizes}")
    print()
    print(
        "batch | pooled mean between | P<=20 | P<=50 | P<=100 | "
        "P<=250 | P<=500 | P<=1000 | fallback rate"
    )
    for row in summary["rows"]:
        print(
            f"{row['recycle_batch_size']} | "
            f"{row['pooled_mean_cards_between']:.2f} | "
            f"{row['pooled_tail_le_20']:.4f} | "
            f"{row['pooled_tail_le_50']:.4f} | "
            f"{row['pooled_tail_le_100']:.4f} | "
            f"{row['pooled_tail_le_250']:.4f} | "
            f"{row['pooled_tail_le_500']:.4f} | "
            f"{row['pooled_tail_le_1000']:.4f} | "
            f"{row['fallback_ejection_rate']:.4%}"
        )
    print()
    print("Aggregate files:")
    print(f"- {config.output_dir / 'summary.json'}")
    print(f"- {config.output_dir / 'summary.csv'}")
    print(f"- {config.output_dir / 'summary.md'}")
    for path in summary["plot_paths"].values():
        print(f"- {config.output_dir / path}")


if __name__ == "__main__":
    main()
