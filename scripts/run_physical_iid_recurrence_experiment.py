# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the six-deck physical IID recurrence experiment."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.physical_iid import (  # noqa: E402
    PhysicalIidRecurrenceConfig,
    run_physical_iid_recurrence_experiment,
)


def parse_args() -> tuple[PhysicalIidRecurrenceConfig, bool]:
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1_000_000)
    parser.add_argument("--deck-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/physical_iid_6deck_1m_seed42"),
    )
    parser.add_argument("--show-histogram-summary", action="store_true")
    args = parser.parse_args()
    return (
        PhysicalIidRecurrenceConfig(
            draws=args.draws,
            deck_count=args.deck_count,
            seed=args.seed,
            output_dir=args.output_dir,
        ),
        args.show_histogram_summary,
    )


def main() -> None:
    """Run the experiment and print a concise statistical table."""
    config, show_histogram_summary = parse_args()
    metrics = run_physical_iid_recurrence_experiment(config)

    print(f"Output directory: {config.output_dir}")
    print(f"Draws: {config.draws}")
    print(f"Deck count: {config.deck_count}")
    print(f"Physical cards: {metrics['physical_card_count']}")
    print(f"Specific-card probability: {metrics['probability']:.8f}")
    print()
    _print_summary_table(metrics)
    if show_histogram_summary:
        print()
        _print_histogram_summary(metrics)
    print()
    print("Generated files:")
    print(f"- {config.output_dir / 'metrics.json'}")
    for target in config.target_physical_ids:
        safe_target = target.replace(":", "_").replace("-", "_")
        print(f"- {config.output_dir / f'target_recurrence_{safe_target}.png'}")
    print(f"- {config.output_dir / 'pooled_physical_recurrence.png'}")


def _print_summary_table(metrics: dict[str, Any]) -> None:
    rows = [
        ("target physical ten", metrics["target_physical_ids"][0]),
        ("target physical five", metrics["target_physical_ids"][1]),
        ("pooled all cards", "pooled_recurrence"),
    ]
    print(
        "label | appearances | returns | mean gap | mean between | "
        "median between | <=100 | <=500"
    )
    for label, key in rows:
        summary = (
            metrics["pooled_recurrence"]
            if key == "pooled_recurrence"
            else metrics["target_recurrence"][key]
        )
        print(
            f"{label} | "
            f"{summary['appearances']} | "
            f"{summary['return_observations']} | "
            f"{_format_optional(summary['mean_draw_gap'])} | "
            f"{_format_optional(summary['mean_cards_between'])} | "
            f"{summary['median_cards_between']} | "
            f"{summary['tail_probabilities']['observed']['100']:.4f} | "
            f"{summary['tail_probabilities']['observed']['500']:.4f}"
        )


def _print_histogram_summary(metrics: dict[str, Any]) -> None:
    for target in metrics["target_physical_ids"]:
        summary = metrics["target_recurrence"][target]
        print(f"{target} histogram bins: {len(summary['cards_between_histogram'])}")
    pooled = metrics["pooled_recurrence"]
    print(f"pooled histogram bins: {len(pooled['cards_between_histogram'])}")


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
