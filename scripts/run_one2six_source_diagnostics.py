# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a direct One2Six card-source diagnostic."""

from argparse import ArgumentParser

from shufflemaster_sim.card_sources import One2SixCardSource, One2SixConfig


def run_diagnostics(
    *,
    draws: int,
    seed: int | None,
    recycle_every: int,
    config: One2SixConfig,
) -> dict[str, object]:
    """Draw and recycle cards through the One2Six source."""
    source = One2SixCardSource(config=config, seed=seed)
    drawn_batch = []
    seen_physical_ids: set[str] = set()

    for _ in range(draws):
        card = source.draw_card()
        drawn_batch.append(card)
        seen_physical_ids.add(card.physical_id)
        if len(drawn_batch) >= recycle_every:
            source.accept_discards(drawn_batch)
            drawn_batch = []

    if drawn_batch:
        source.accept_discards(drawn_batch)

    source.assert_invariants()
    occupancy = source.carousel_occupancy
    return {
        "draws": draws,
        "unique_physical_ids_seen": len(seen_physical_ids),
        "ejection_count": source.ejection_count,
        "fallback_ejection_count": source.fallback_ejection_count,
        "final_buffer_size": source.output_buffer_size,
        "carousel_min_occupancy": min(occupancy),
        "carousel_max_occupancy": max(occupancy),
        "carousel_total_cards": sum(occupancy),
        "invariant_check": "passed",
    }


def parse_args() -> tuple[int, int | None, int, One2SixConfig]:
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--recycle-every", type=int, default=52)
    parser.add_argument("--deck-count", type=int, default=6)
    parser.add_argument("--carousel-slots", type=int, default=38)
    parser.add_argument("--slot-capacity", type=int, default=10)
    parser.add_argument("--buffer-target", type=int, default=18)
    parser.add_argument("--refill-threshold", type=int, default=8)
    parser.add_argument("--min-ejection-cards", type=int, default=7)
    args = parser.parse_args()
    config = One2SixConfig(
        deck_count=args.deck_count,
        carousel_slot_count=args.carousel_slots,
        slot_capacity=args.slot_capacity,
        output_buffer_target=args.buffer_target,
        refill_threshold=args.refill_threshold,
        min_cards_for_ejection=args.min_ejection_cards,
    )
    return args.draws, args.seed, args.recycle_every, config


def main() -> None:
    """Run diagnostics and print a compact summary."""
    draws, seed, recycle_every, config = parse_args()
    result = run_diagnostics(
        draws=draws,
        seed=seed,
        recycle_every=recycle_every,
        config=config,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
