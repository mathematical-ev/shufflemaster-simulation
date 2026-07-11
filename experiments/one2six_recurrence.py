# SPDX-License-Identifier: GPL-3.0-or-later

"""One2Six-style physical-card recurrence experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import median
from typing import Any

from experiments.plots import plot_physical_recurrence_histogram
from shufflemaster_sim.card_sources import One2SixCardSource, One2SixConfig
from shufflemaster_sim.cards import Card

DEFAULT_ONE2SIX_TEN_SPADES = "one2six-deck-0:T:spades"
DEFAULT_ONE2SIX_FIVE_SPADES = "one2six-deck-0:5:spades"
TAIL_LIMITS = (20, 50, 100, 250, 500, 1000)
GOODNESS_OF_FIT_BINS = (
    (0, 20),
    (21, 50),
    (51, 100),
    (101, 250),
    (251, 500),
    (501, 1000),
    (1001, 2000),
    (2001, None),
)


@dataclass(frozen=True, slots=True)
class One2SixRecurrenceConfig:
    """Configuration for One2Six-style recurrence experiments."""

    draws: int = 1_000_000
    recycle_batch_size: int = 20
    seed: int | None = 42
    output_dir: Path = Path("experiments/outputs/one2six_recurrence_1m_seed42")
    target_physical_ids: tuple[str, ...] = (
        DEFAULT_ONE2SIX_TEN_SPADES,
        DEFAULT_ONE2SIX_FIVE_SPADES,
    )
    one2six_config: One2SixConfig | None = None

    def __post_init__(self) -> None:
        if self.draws <= 0:
            raise ValueError("draws must be positive.")
        if self.recycle_batch_size <= 0:
            raise ValueError("recycle_batch_size must be positive.")


def run_one2six_recurrence_experiment(
    config: One2SixRecurrenceConfig,
) -> dict[str, Any]:
    """Run the One2Six recurrence experiment and write outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_one2six_recurrence_metrics(config)
    _write_one2six_plots(config, metrics)
    _write_json(config.output_dir / "metrics.json", metrics)
    return metrics


def calculate_one2six_recurrence_metrics(
    config: One2SixRecurrenceConfig,
) -> dict[str, Any]:
    """Calculate target and pooled recurrence metrics for One2Six draws."""
    one2six_config = config.one2six_config or One2SixConfig()
    source = One2SixCardSource(config=one2six_config, seed=config.seed)
    target_set = set(config.target_physical_ids)
    known_ids = _known_physical_ids(source)
    missing_targets = target_set - known_ids
    if missing_targets:
        missing = ", ".join(sorted(missing_targets))
        raise ValueError(f"Unknown target physical ids: {missing}")

    target_positions: dict[str, list[int]] = {
        target: [] for target in config.target_physical_ids
    }
    target_gaps: dict[str, list[int]] = {
        target: [] for target in config.target_physical_ids
    }
    last_seen_by_physical_id: dict[str, int] = {}
    pooled_gaps: list[int] = []
    appearances_by_physical_id = {physical_id: 0 for physical_id in known_ids}
    return_counts_by_physical_id = {physical_id: 0 for physical_id in known_ids}
    recycle_batch: list[Card] = []
    accepted_batches: list[AcceptedBatch] = []

    for draw_index in range(config.draws):
        card = source.draw_card()
        appearances_by_physical_id[card.physical_id] += 1
        if card.physical_id in target_set:
            target_positions[card.physical_id].append(draw_index)
        previous_draw_index = last_seen_by_physical_id.get(card.physical_id)
        if previous_draw_index is not None:
            draw_gap = draw_index - previous_draw_index
            pooled_gaps.append(draw_gap)
            return_counts_by_physical_id[card.physical_id] += 1
            if card.physical_id in target_set:
                target_gaps[card.physical_id].append(draw_gap)
        last_seen_by_physical_id[card.physical_id] = draw_index

        recycle_batch.append(card)
        if len(recycle_batch) >= config.recycle_batch_size:
            accepted_batches.append(
                AcceptedBatch(
                    batch_index=len(accepted_batches),
                    accepted_after_draw_index=draw_index,
                    physical_ids=[card.physical_id for card in recycle_batch],
                )
            )
            source.accept_discards(recycle_batch)
            recycle_batch = []

    if recycle_batch:
        accepted_batches.append(
            AcceptedBatch(
                batch_index=len(accepted_batches),
                accepted_after_draw_index=config.draws - 1,
                physical_ids=[card.physical_id for card in recycle_batch],
            )
        )
        source.accept_discards(recycle_batch)

    external_cards: list[Card] = []
    source.assert_invariants(external_cards=external_cards)
    physical_card_count = source.cards_total_known
    physical_iid_probability = 1.0 / physical_card_count
    target_metrics = {
        target: recurrence_summary(
            total_draws=config.draws,
            appearances=len(target_positions[target]),
            draw_gaps=target_gaps[target],
            physical_iid_probability=physical_iid_probability,
        )
        for target in config.target_physical_ids
    }
    for target, summary in target_metrics.items():
        summary["positions"] = target_positions[target]

    telemetry = source.telemetry_records()
    source_diagnostics = one2six_source_diagnostics(
        source=source,
        telemetry=telemetry,
        accepted_batches=accepted_batches,
    )
    return {
        "experiment_name": "one2six_physical_recurrence",
        "model": "one2six",
        "card_source_type": "one2six",
        "config": {
            "draws": config.draws,
            "recycle_batch_size": config.recycle_batch_size,
            "seed": config.seed,
            "output_dir": str(config.output_dir),
            "target_physical_ids": list(config.target_physical_ids),
            "one2six_config": one2six_config_as_dict(one2six_config),
        },
        "draw_index_base": 0,
        "draws": config.draws,
        "recycle_batch_size": config.recycle_batch_size,
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "physical_card_count": physical_card_count,
        "physical_iid_comparator_probability": physical_iid_probability,
        "physical_iid_comparator_model": (
            "Six-deck physical IID comparator: P(cards_between = n) = "
            "(1 - p) ** n * p with p = 1 / physical_card_count."
        ),
        "target_physical_ids": list(config.target_physical_ids),
        "target_recurrence": target_metrics,
        "pooled_recurrence": recurrence_summary(
            total_draws=config.draws,
            appearances=sum(appearances_by_physical_id.values()),
            draw_gaps=pooled_gaps,
            physical_iid_probability=physical_iid_probability,
        ),
        "per_physical_card_diagnostics": count_diagnostics(
            appearance_counts=list(appearances_by_physical_id.values()),
            return_counts=list(return_counts_by_physical_id.values()),
        ),
        "one2six_source_diagnostics": source_diagnostics,
        "plot_paths": plot_paths(config),
        "assumptions_and_caveats": [
            "Current One2Six model is a configurable reconstruction, not a "
            "verified proprietary-device model.",
            "Whole selected shelf ejection means eject the entire selected shelf, "
            "not necessarily a shelf at capacity.",
            "Fallback ejection can eject shelves below min_cards_for_ejection.",
            "Accepted discards are ingested immediately under the default policy.",
            "Returned cards become drawable only after shelf ejection into the "
            "output buffer and subsequent buffer draw.",
            "Recurrence deviations are not exploitability evidence by themselves.",
        ],
    }


@dataclass(frozen=True, slots=True)
class AcceptedBatch:
    """Accepted discard batch metadata for source-level diagnostics."""

    batch_index: int
    accepted_after_draw_index: int
    physical_ids: list[str]


def recurrence_summary(
    *,
    total_draws: int,
    appearances: int,
    draw_gaps: list[int],
    physical_iid_probability: float,
) -> dict[str, Any]:
    """Summarize empirical One2Six recurrence against physical IID comparator."""
    cards_between = [draw_gap - 1 for draw_gap in draw_gaps]
    return_observations = len(draw_gaps)
    return {
        "total_draws": total_draws,
        "appearances": appearances,
        "return_observations": return_observations,
        "observed_appearance_rate": appearances / total_draws if total_draws else 0.0,
        "physical_iid_expected_appearance_rate": physical_iid_probability,
        "mean_draw_gap": mean(draw_gaps),
        "physical_iid_expected_mean_draw_gap": 1.0 / physical_iid_probability,
        "mean_cards_between": mean(cards_between),
        "physical_iid_expected_mean_cards_between": (
            (1.0 - physical_iid_probability) / physical_iid_probability
        ),
        "median_cards_between": median(cards_between) if cards_between else None,
        "min_cards_between": min(cards_between) if cards_between else None,
        "max_cards_between": max(cards_between) if cards_between else None,
        "quantiles": quantiles(cards_between, (0.5, 0.75, 0.9, 0.95, 0.99)),
        "tail_probabilities": {
            "observed": {
                str(limit): tail_probability(cards_between, limit)
                for limit in TAIL_LIMITS
            },
            "physical_iid_theoretical": {
                str(limit): theoretical_tail_probability(
                    physical_iid_probability,
                    limit,
                )
                for limit in TAIL_LIMITS
            },
        },
        "cards_between_histogram": dict(sorted(histogram(cards_between).items())),
        "goodness_of_fit": binned_goodness_of_fit(
            cards_between=cards_between,
            probability=physical_iid_probability,
        ),
    }


def one2six_source_diagnostics(
    *,
    source: One2SixCardSource,
    telemetry: list[dict[str, object]],
    accepted_batches: list[AcceptedBatch],
) -> dict[str, Any]:
    """Return One2Six source diagnostics for recurrence interpretation."""
    ejection_records = source.ejection_records()
    ejection_group_sizes = [int(record["group_size"]) for record in ejection_records]
    occupancy = list(source.carousel_occupancy)
    fallback_count = source.fallback_ejection_count
    ejection_count = source.ejection_count
    return {
        "one2six_config": one2six_config_as_dict(source.config),
        "output_buffer_size_at_end": source.output_buffer_size,
        "final_output_buffer_size": source.output_buffer_size,
        "ejection_count": ejection_count,
        "fallback_ejection_count": fallback_count,
        "fallback_ejection_rate": fallback_count / ejection_count
        if ejection_count
        else 0.0,
        "ejection_group_size_distribution": dict(
            sorted(histogram(ejection_group_sizes).items())
        ),
        "ejection_group_size_summary": distribution_summary(ejection_group_sizes),
        "final_carousel_occupancy_summary": distribution_summary(occupancy),
        "accepted_discard_batch_count": source.accepted_discard_batch_count,
        "invariant_check": invariant_status(source),
        "telemetry_latency_summary": telemetry_latency_summary(telemetry),
        "discard_batch_reappearance": discard_batch_reappearance_summary(
            telemetry=telemetry,
            accepted_batches=accepted_batches,
        ),
    }


def telemetry_latency_summary(
    telemetry: list[dict[str, object]],
) -> dict[str, Any]:
    """Summarize clean latency intervals available from One2Six telemetry."""
    last_carousel_entry: dict[str, int] = {}
    last_output_entry: dict[str, int] = {}
    last_accept: dict[str, int] = {}
    shelf_residence: list[int] = []
    buffer_latency: list[int] = []
    accepted_to_redraw: list[int] = []

    for record in telemetry:
        physical_id = str(record["physical_id"])
        event_sequence = int(record["event_sequence"])
        event_type = record["event_type"]
        if event_type == "accepted_discard":
            last_accept[physical_id] = event_sequence
        elif event_type == "entered_carousel_shelf":
            last_carousel_entry[physical_id] = event_sequence
        elif event_type == "entered_output_buffer":
            carousel_sequence = last_carousel_entry.get(physical_id)
            if carousel_sequence is not None:
                shelf_residence.append(event_sequence - carousel_sequence)
            last_output_entry[physical_id] = event_sequence
        elif event_type == "drawn_from_output_buffer":
            output_sequence = last_output_entry.get(physical_id)
            if output_sequence is not None:
                buffer_latency.append(event_sequence - output_sequence)
            accept_sequence = last_accept.get(physical_id)
            if accept_sequence is not None:
                accepted_to_redraw.append(event_sequence - accept_sequence)
                del last_accept[physical_id]

    return {
        "shelf_residence_event_delta": distribution_summary(shelf_residence),
        "buffer_latency_event_delta": distribution_summary(buffer_latency),
        "accepted_to_redraw_event_delta": distribution_summary(accepted_to_redraw),
        "limitations": (
            "Latencies are measured in telemetry event-sequence deltas, not draw "
            "counts. Draw-count latency is captured for accepted-batch first "
            "reappearance."
        ),
    }


def discard_batch_reappearance_summary(
    *,
    telemetry: list[dict[str, object]],
    accepted_batches: list[AcceptedBatch],
) -> dict[str, Any]:
    """Summarize first accepted-batch reappearance in draw-count units."""
    first_draw_after_accept: dict[int, int | None] = {
        batch.batch_index: None for batch in accepted_batches
    }
    pending_batches_by_card: defaultdict[str, list[AcceptedBatch]] = defaultdict(list)
    for batch in accepted_batches:
        for physical_id in batch.physical_ids:
            pending_batches_by_card[physical_id].append(batch)

    for record in telemetry:
        if record["event_type"] != "drawn_from_output_buffer":
            continue
        draw_id = int(record["draw_id"])
        physical_id = str(record["physical_id"])
        pending = pending_batches_by_card.get(physical_id, [])
        remaining = []
        for batch in pending:
            if draw_id > batch.accepted_after_draw_index:
                current = first_draw_after_accept[batch.batch_index]
                delta = draw_id - batch.accepted_after_draw_index
                if current is None or delta < current:
                    first_draw_after_accept[batch.batch_index] = delta
            else:
                remaining.append(batch)
        pending_batches_by_card[physical_id] = remaining

    observed_deltas = [
        delta for delta in first_draw_after_accept.values() if delta is not None
    ]
    thresholds = (20, 50, 100, 250, 500, 1000)
    return {
        "accepted_batch_count": len(accepted_batches),
        "batches_with_reappearance": len(observed_deltas),
        "first_reappearance_draw_delta_summary": distribution_summary(observed_deltas),
        "within_threshold_counts": {
            str(threshold): sum(1 for delta in observed_deltas if delta <= threshold)
            for threshold in thresholds
        },
        "within_threshold_rates": {
            str(threshold): (
                sum(1 for delta in observed_deltas if delta <= threshold)
                / len(accepted_batches)
                if accepted_batches
                else 0.0
            )
            for threshold in thresholds
        },
    }


def binned_goodness_of_fit(
    *,
    cards_between: list[int],
    probability: float,
) -> dict[str, Any]:
    """Return binned observed-vs-physical-IID recurrence diagnostics."""
    total_observations = len(cards_between)
    counts = histogram(cards_between)
    bins = []
    chi_square_statistic = 0.0
    for lower, upper in GOODNESS_OF_FIT_BINS:
        observed = observed_bin_count(counts, lower, upper)
        expected_probability = geometric_bin_probability(probability, lower, upper)
        expected = total_observations * expected_probability
        residual = observed - expected
        standardized_residual = residual / sqrt(expected) if expected > 0 else None
        if expected > 0:
            chi_square_statistic += (residual * residual) / expected
        bins.append(
            {
                "label": bin_label(lower, upper),
                "lower": lower,
                "upper": upper,
                "observed_count": observed,
                "physical_iid_expected_count": expected,
                "residual": residual,
                "standardized_residual": standardized_residual,
            }
        )
    return {
        "description": (
            "Binned observed-vs-physical-IID diagnostic; chi_square_statistic "
            "is descriptive only and is not a formal p-value."
        ),
        "bins": bins,
        "chi_square_statistic": chi_square_statistic,
    }


def count_diagnostics(
    *,
    appearance_counts: list[int],
    return_counts: list[int],
) -> dict[str, Any]:
    """Return compact per-card count diagnostics."""
    return {
        "appearance_counts": distribution_summary_with_histogram(appearance_counts),
        "return_counts": distribution_summary_with_histogram(return_counts),
    }


def distribution_summary_with_histogram(values: list[int]) -> dict[str, Any]:
    """Return summary statistics and compact histogram for integer counts."""
    summary = distribution_summary(values)
    summary["distribution"] = dict(sorted(histogram(values).items()))
    return summary


def distribution_summary(values: list[int]) -> dict[str, Any]:
    """Return compact summary statistics for integer values."""
    return {
        "min": min(values) if values else None,
        "mean": mean(values),
        "median": median(values) if values else None,
        "max": max(values) if values else None,
        "quantiles": quantiles(values, (0.05, 0.25, 0.5, 0.75, 0.95)),
    }


def one2six_config_as_dict(config: One2SixConfig) -> dict[str, object]:
    """Return JSON-friendly One2Six config fields."""
    return {
        "deck_count": config.deck_count,
        "carousel_slot_count": config.carousel_slot_count,
        "slot_capacity": config.slot_capacity,
        "output_buffer_target": config.output_buffer_target,
        "refill_threshold": config.refill_threshold,
        "min_cards_for_ejection": config.min_cards_for_ejection,
        "insertion_rule": config.insertion_rule,
        "output_selection_rule": config.output_selection_rule,
        "ejection_rule": config.ejection_rule,
        "input_feed_order": config.input_feed_order,
        "accepted_cards_orientation": config.accepted_cards_orientation,
        "intra_slot_order": config.intra_slot_order,
        "ejected_group_order": config.ejected_group_order,
        "ingest_policy": config.ingest_policy,
        "fallback_when_no_eligible_slot": config.fallback_when_no_eligible_slot,
        "strict_invariants": config.strict_invariants,
    }


def plot_paths(config: One2SixRecurrenceConfig) -> dict[str, Any]:
    """Return generated plot paths relative to output directory."""
    return {
        "target_recurrence": {
            target: f"target_recurrence_{safe_name(target)}.png"
            for target in config.target_physical_ids
        },
        "pooled_one2six_physical_recurrence": (
            "pooled_one2six_physical_recurrence.png"
        ),
        "one2six_vs_physical_iid_pooled_recurrence": (
            "one2six_vs_physical_iid_pooled_recurrence.png"
        ),
    }


def _write_one2six_plots(
    config: One2SixRecurrenceConfig,
    metrics: dict[str, Any],
) -> None:
    probability = metrics["physical_iid_comparator_probability"]
    for target, summary in metrics["target_recurrence"].items():
        plot_physical_recurrence_histogram(
            summary,
            probability=probability,
            title=f"One2Six Recurrence: {target}",
            output_path=config.output_dir
            / f"target_recurrence_{safe_name(target)}.png",
        )
    plot_physical_recurrence_histogram(
        metrics["pooled_recurrence"],
        probability=probability,
        title="Pooled One2Six Physical Recurrence",
        output_path=config.output_dir / "pooled_one2six_physical_recurrence.png",
    )
    plot_physical_recurrence_histogram(
        metrics["pooled_recurrence"],
        probability=probability,
        title="One2Six vs Physical IID Pooled Recurrence",
        output_path=config.output_dir / "one2six_vs_physical_iid_pooled_recurrence.png",
    )


def invariant_status(source: One2SixCardSource) -> str:
    """Return invariant check status without raising from diagnostics."""
    try:
        source.assert_invariants()
    except AssertionError as exc:
        return f"failed: {exc}"
    return "passed"


def mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def quantiles(
    values: list[int],
    requested_quantiles: tuple[float, ...],
) -> dict[str, int | None]:
    if not values:
        return {f"{quantile:.2f}": None for quantile in requested_quantiles}
    sorted_values = sorted(values)
    last_index = len(sorted_values) - 1
    return {
        f"{quantile:.2f}": sorted_values[round(last_index * quantile)]
        for quantile in requested_quantiles
    }


def tail_probability(values: list[int], limit: int) -> float:
    return sum(1 for value in values if value <= limit) / len(values) if values else 0.0


def theoretical_tail_probability(probability: float, limit: int) -> float:
    return 1.0 - ((1.0 - probability) ** (limit + 1))


def histogram(values: list[int]) -> dict[int, int]:
    counts: defaultdict[int, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(counts)


def observed_bin_count(
    counts: dict[int, int],
    lower: int,
    upper: int | None,
) -> int:
    if upper is None:
        return sum(count for value, count in counts.items() if value >= lower)
    return sum(count for value, count in counts.items() if lower <= value <= upper)


def geometric_bin_probability(
    probability: float,
    lower: int,
    upper: int | None,
) -> float:
    lower_tail_before = 1.0 - ((1.0 - probability) ** lower)
    if upper is None:
        return 1.0 - lower_tail_before
    return theoretical_tail_probability(probability, upper) - lower_tail_before


def bin_label(lower: int, upper: int | None) -> str:
    return f"{lower}+" if upper is None else f"{lower}-{upper}"


def safe_name(value: str) -> str:
    return value.replace(":", "_").replace("-", "_")


def _known_physical_ids(source: One2SixCardSource) -> set[str]:
    # Public enough for experiments via cards_total_known, but IDs are currently
    # only exposed through the implementation's invariant set.
    return set(source._known_physical_ids)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
