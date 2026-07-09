"""Run the One2Six physical-card recurrence experiment."""

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.one2six_recurrence import (  # noqa: E402
    One2SixRecurrenceConfig,
    run_one2six_recurrence_experiment,
)

from shufflemaster_sim.card_sources import One2SixConfig  # noqa: E402


def parse_args() -> One2SixRecurrenceConfig:
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1_000_000)
    parser.add_argument("--recycle-batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/one2six_recurrence_1m_seed42"),
    )
    parser.add_argument("--deck-count", type=int, default=6)
    parser.add_argument("--carousel-slots", type=int, default=38)
    parser.add_argument("--slot-capacity", type=int, default=10)
    parser.add_argument("--buffer-target", type=int, default=18)
    parser.add_argument("--refill-threshold", type=int, default=8)
    parser.add_argument("--min-ejection-cards", type=int, default=7)
    args = parser.parse_args()
    return One2SixRecurrenceConfig(
        draws=args.draws,
        recycle_batch_size=args.recycle_batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        one2six_config=One2SixConfig(
            deck_count=args.deck_count,
            carousel_slot_count=args.carousel_slots,
            slot_capacity=args.slot_capacity,
            output_buffer_target=args.buffer_target,
            refill_threshold=args.refill_threshold,
            min_cards_for_ejection=args.min_ejection_cards,
        ),
    )


def main() -> None:
    """Run the experiment and print a compact summary."""
    config = parse_args()
    metrics = run_one2six_recurrence_experiment(config)
    diagnostics = metrics["one2six_source_diagnostics"]

    print(f"Output directory: {config.output_dir}")
    print(f"Draws: {config.draws}")
    print(f"Recycle batch size: {config.recycle_batch_size}")
    print(f"Physical cards: {metrics['physical_card_count']}")
    print(
        "Physical IID comparator probability: "
        f"{metrics['physical_iid_comparator_probability']:.8f}"
    )
    print()
    print_summary_table(metrics)
    print()
    print(f"Ejection count: {diagnostics['ejection_count']}")
    print(f"Fallback ejection count: {diagnostics['fallback_ejection_count']}")
    print(f"Fallback ejection rate: {diagnostics['fallback_ejection_rate']:.4%}")
    print(f"Final output buffer size: {diagnostics['final_output_buffer_size']}")
    print(f"Invariant check: {diagnostics['invariant_check']}")
    print()
    print("Generated files:")
    print(f"- {config.output_dir / 'metrics.json'}")
    plot_paths = metrics["plot_paths"]
    for path in plot_paths["target_recurrence"].values():
        print(f"- {config.output_dir / path}")
    print(f"- {config.output_dir / plot_paths['pooled_one2six_physical_recurrence']}")
    comparison_plot = plot_paths["one2six_vs_physical_iid_pooled_recurrence"]
    print(f"- {config.output_dir / comparison_plot}")


def print_summary_table(metrics: dict[str, Any]) -> None:
    """Print target and pooled recurrence summary rows."""
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
            f"{format_optional(summary['mean_draw_gap'])} | "
            f"{format_optional(summary['mean_cards_between'])} | "
            f"{summary['median_cards_between']} | "
            f"{summary['tail_probabilities']['observed']['100']:.4f} | "
            f"{summary['tail_probabilities']['observed']['500']:.4f}"
        )


def format_optional(value: float | None) -> str:
    """Format optional floats for compact CLI output."""
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
