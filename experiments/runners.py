# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Experiment runners for controlled source and game validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.metrics import (
    game_metrics_from_result,
    source_draw_metrics,
    target_card_label,
)
from experiments.plots import (
    plot_cumulative_profit,
    plot_interarrival_histogram,
    plot_outcome_percentages,
    plot_signed_streak_histogram,
    plot_signed_streak_smoothed_density,
)
from shufflemaster_sim.card_sources import IidRandomCardSource
from shufflemaster_sim.cards import Card
from shufflemaster_sim.simulation import SimulationConfig, run_casino_blackjack_baseline


@dataclass(frozen=True, slots=True)
class IidBaselineExperimentConfig:
    """Configuration for IID source and Casino Blackjack baseline experiments."""

    source_draws: int = 1_000_000
    game_rounds: int = 1_000_000
    base_bet: float = 10.0
    seed: int | None = 42
    output_dir: Path = Path("experiments/outputs/iid_baseline")
    target_cards: tuple[str, ...] = ("T:S", "5:S")
    rank_targets: tuple[str, ...] = ("T", "5")
    run_source_experiment: bool = True
    run_game_experiment: bool = True
    save_raw: bool = False

    def __post_init__(self) -> None:
        if self.source_draws <= 0:
            raise ValueError("source_draws must be positive.")
        if self.game_rounds <= 0:
            raise ValueError("game_rounds must be positive.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        if not self.run_source_experiment and not self.run_game_experiment:
            raise ValueError("At least one experiment must be enabled.")


def run_iid_source_experiment(config: IidBaselineExperimentConfig) -> dict[str, Any]:
    """Run IID source draws and return aggregate metrics."""
    source = IidRandomCardSource(seed=config.seed)
    cards = [source.draw_card() for _ in range(config.source_draws)]
    metrics = source_draw_metrics(
        cards,
        target_cards=config.target_cards,
        rank_targets=config.rank_targets,
    )
    if config.save_raw:
        _write_source_raw_cards(cards, config.output_dir / "source_draws.jsonl")
    return metrics


def run_iid_game_experiment(config: IidBaselineExperimentConfig) -> dict[str, Any]:
    """Run Casino Blackjack with IID cards and return aggregate metrics."""
    result = run_casino_blackjack_baseline(
        SimulationConfig(
            rounds=config.game_rounds,
            base_bet=config.base_bet,
            seed=config.seed,
            card_source="iid",
        )
    )
    metrics = game_metrics_from_result(result)
    if config.save_raw:
        _write_jsonl(
            config.output_dir / "game_rounds.jsonl",
            result.as_round_records(),
        )
    return metrics


def run_iid_baseline_experiment(
    config: IidBaselineExperimentConfig,
) -> dict[str, Any]:
    """Run configured IID baseline experiments and write outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_metrics: dict[str, Any] | None = None
    game_metrics: dict[str, Any] | None = None
    plot_paths: list[str] = []

    if config.run_source_experiment:
        source_metrics = run_iid_source_experiment(config)
        _write_json(config.output_dir / "source_metrics.json", source_metrics)
        plot_paths.extend(_write_source_plots(config, source_metrics))

    if config.run_game_experiment:
        game_metrics = run_iid_game_experiment(config)
        _write_json(config.output_dir / "game_metrics.json", game_metrics)
        plot_paths.extend(_write_game_plots(config, game_metrics))

    metrics = {
        "config": {
            "source_draws": config.source_draws,
            "game_rounds": config.game_rounds,
            "base_bet": config.base_bet,
            "seed": config.seed,
            "target_cards": list(config.target_cards),
            "rank_targets": list(config.rank_targets),
            "run_source_experiment": config.run_source_experiment,
            "run_game_experiment": config.run_game_experiment,
            "save_raw": config.save_raw,
        },
        "source_metrics": source_metrics,
        "game_metrics": game_metrics,
        "plot_paths": plot_paths,
    }
    _write_json(config.output_dir / "metrics.json", metrics)
    return metrics


def _write_source_plots(
    config: IidBaselineExperimentConfig,
    source_metrics: dict[str, Any],
) -> list[str]:
    plot_paths: list[str] = []
    target_metrics = source_metrics["target_card_recurrence"]
    for target in config.target_cards:
        path = (
            config.output_dir
            / f"target_card_interarrival_{target_card_label(target)}.png"
        )
        plot_paths.append(
            str(
                plot_interarrival_histogram(
                    target_metrics[target],
                    probability=1.0 / 52.0,
                    title=f"Target Card Inter-arrival: {target}",
                    output_path=path,
                )
            )
        )

    rank_metrics = source_metrics["rank_target_recurrence"]
    for rank in config.rank_targets:
        path = config.output_dir / f"rank_interarrival_{rank}.png"
        plot_paths.append(
            str(
                plot_interarrival_histogram(
                    rank_metrics[rank],
                    probability=1.0 / 13.0,
                    title=f"Rank Inter-arrival: {rank}",
                    output_path=path,
                )
            )
        )
    return plot_paths


def _write_game_plots(
    config: IidBaselineExperimentConfig,
    game_metrics: dict[str, Any],
) -> list[str]:
    plot_paths = [
        plot_signed_streak_histogram(
            game_metrics["streaks"],
            theoretical=game_metrics["theoretical_streak_probabilities"],
            output_path=config.output_dir / "streak_histogram.png",
        ),
        plot_signed_streak_smoothed_density(
            game_metrics["streaks"],
            output_path=config.output_dir / "streak_smoothed_density.png",
        ),
        plot_outcome_percentages(
            game_metrics,
            output_path=config.output_dir / "outcome_percentages.png",
        ),
        plot_cumulative_profit(
            game_metrics["cumulative_profit_path"],
            output_path=config.output_dir / "cumulative_profit.png",
        ),
    ]
    return [str(path) for path in plot_paths]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True))
            output.write("\n")


def _write_source_raw_cards(cards: list[Card], path: Path) -> None:
    records = [
        {
            "draw_id": card.draw_id,
            "physical_id": card.physical_id,
            "rank": card.rank,
            "suit": card.suit,
        }
        for card in cards
    ]
    _write_jsonl(path, records)
