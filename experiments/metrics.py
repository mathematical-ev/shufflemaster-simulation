"""Metrics for experiment-level validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from math import exp, pi, sqrt
from statistics import median
from typing import Any

from shufflemaster_sim.cards import RANKS, SUITS, Card, Rank, Suit

TEN_VALUE_RANKS = frozenset({"T", "J", "Q", "K"})
LOW_RANKS = frozenset({"2", "3", "4", "5", "6"})
NEUTRAL_RANKS = frozenset({"7", "8", "9"})
HIGH_RANKS = TEN_VALUE_RANKS | {"A"}
SUIT_ALIASES: dict[str, Suit] = {
    "C": "clubs",
    "D": "diamonds",
    "H": "hearts",
    "S": "spades",
    "clubs": "clubs",
    "diamonds": "diamonds",
    "hearts": "hearts",
    "spades": "spades",
}


def hilo_value(rank: Rank) -> int:
    """Return the Hi-Lo value for a rank."""
    if rank in LOW_RANKS:
        return 1
    if rank in NEUTRAL_RANKS:
        return 0
    return -1


def parse_target_card(target: str) -> tuple[Rank, Suit]:
    """Parse target card notation such as 'T:S' or 'T_spades'."""
    normalized = target.strip().replace("_", ":")
    if ":" in normalized:
        rank_text, suit_text = normalized.split(":", maxsplit=1)
    else:
        rank_text = normalized[0]
        suit_text = normalized[1:]
    rank = rank_text.upper()
    suit_key = suit_text.strip()
    if rank not in RANKS:
        raise ValueError(f"Unknown rank in target card: {target}")
    if suit_key not in SUIT_ALIASES:
        raise ValueError(f"Unknown suit in target card: {target}")
    return rank, SUIT_ALIASES[suit_key]


def target_card_label(target: str) -> str:
    """Return a filename-friendly target-card label."""
    rank, suit = parse_target_card(target)
    return f"{rank}_{suit[0].upper()}"


def recurrence_from_positions(positions: Sequence[int]) -> dict[str, Any]:
    """Build recurrence metrics from zero-based target positions."""
    gaps = [
        current - previous
        for previous, current in zip(positions, positions[1:], strict=False)
    ]
    cards_between = [gap - 1 for gap in gaps]
    histogram = dict(sorted(Counter(cards_between).items()))
    return {
        "positions": list(positions),
        "gaps": gaps,
        "cards_between": cards_between,
        "observed_mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "observed_median_gap": median(gaps) if gaps else 0.0,
        "min_gap": min(gaps) if gaps else None,
        "max_gap": max(gaps) if gaps else None,
        "cards_between_histogram": histogram,
    }


def geometric_probabilities(
    max_cards_between: int,
    probability: float,
) -> dict[int, float]:
    """Return IID geometric probabilities for cards-between support."""
    return {
        n: ((1.0 - probability) ** n) * probability
        for n in range(max_cards_between + 1)
    }


def fixed_window_counts(
    cards: Sequence[Card],
    *,
    window_size: int,
    predicate: object,
) -> list[int]:
    """Count target appearances in fixed windows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if not callable(predicate):
        raise TypeError("predicate must be callable.")
    return [
        sum(1 for card in cards[index : index + window_size] if predicate(card))
        for index in range(0, len(cards), window_size)
    ]


def source_draw_metrics(
    cards: Sequence[Card],
    *,
    target_cards: Sequence[str],
    rank_targets: Sequence[str],
    rolling_windows: Sequence[int] = (10, 20, 50),
    fixed_window_size: int = 100,
) -> dict[str, Any]:
    """Summarize IID source draws."""
    total_draws = len(cards)
    rank_counts = Counter(card.rank for card in cards)
    suit_counts = Counter(card.suit for card in cards)
    rank_suit_counts = Counter(f"{card.rank}:{card.suit}" for card in cards)
    hilo_values = [hilo_value(card.rank) for card in cards]
    hilo_mean = sum(hilo_values) / total_draws if total_draws else 0.0
    hilo_variance = (
        sum((value - hilo_mean) ** 2 for value in hilo_values) / total_draws
        if total_draws
        else 0.0
    )
    rolling_hilo = {
        str(window): dict(
            sorted(
                Counter(
                    sum(hilo_values[index : index + window])
                    for index in range(0, max(total_draws - window + 1, 0))
                ).items()
            )
        )
        for window in rolling_windows
    }

    target_card_metrics: dict[str, Any] = {}
    for target in target_cards:
        rank, suit = parse_target_card(target)
        positions = [
            index
            for index, card in enumerate(cards)
            if card.rank == rank and card.suit == suit
        ]
        target_card_metrics[target] = {
            **recurrence_from_positions(positions),
            "probability": 1.0 / 52.0,
            "fixed_window_counts": fixed_window_counts(
                cards,
                window_size=fixed_window_size,
                predicate=lambda card, rank=rank, suit=suit: (
                    card.rank == rank and card.suit == suit
                ),
            ),
        }

    rank_target_metrics: dict[str, Any] = {}
    for rank in rank_targets:
        if rank not in RANKS:
            raise ValueError(f"Unknown rank target: {rank}")
        positions = [index for index, card in enumerate(cards) if card.rank == rank]
        rank_target_metrics[rank] = {
            **recurrence_from_positions(positions),
            "probability": 1.0 / 13.0,
            "fixed_window_counts": fixed_window_counts(
                cards,
                window_size=fixed_window_size,
                predicate=lambda card, rank=rank: card.rank == rank,
            ),
        }

    ace_count = rank_counts["A"]
    ten_value_count = sum(rank_counts[rank] for rank in TEN_VALUE_RANKS)
    low_count = sum(rank_counts[rank] for rank in LOW_RANKS)
    neutral_count = sum(rank_counts[rank] for rank in NEUTRAL_RANKS)
    high_count = sum(rank_counts[rank] for rank in HIGH_RANKS)

    return {
        "total_draws": total_draws,
        "rank_counts": {rank: rank_counts[rank] for rank in RANKS},
        "suit_counts": {suit: suit_counts[suit] for suit in SUITS},
        "rank_suit_counts": dict(sorted(rank_suit_counts.items())),
        "ace_count": ace_count,
        "ace_rate": _rate(ace_count, total_draws),
        "ten_value_count": ten_value_count,
        "ten_value_rate": _rate(ten_value_count, total_draws),
        "low_card_count": low_count,
        "low_card_rate": _rate(low_count, total_draws),
        "neutral_count": neutral_count,
        "neutral_rate": _rate(neutral_count, total_draws),
        "high_card_count": high_count,
        "high_card_rate": _rate(high_count, total_draws),
        "hilo_values_seen": sorted(set(hilo_values)),
        "hilo_mean": hilo_mean,
        "hilo_variance": hilo_variance,
        "rolling_hilo_window_distributions": rolling_hilo,
        "target_card_recurrence": target_card_metrics,
        "rank_target_recurrence": rank_target_metrics,
    }


def streak_distributions(outcomes: Iterable[float]) -> dict[str, dict[int, int]]:
    """Return streak distributions where pushes do not break streaks."""
    win_streaks: Counter[int] = Counter()
    loss_streaks: Counter[int] = Counter()
    current_kind: str | None = None
    current_length = 0

    def flush() -> None:
        nonlocal current_kind, current_length
        if current_kind == "win" and current_length > 0:
            win_streaks[current_length] += 1
        elif current_kind == "loss" and current_length > 0:
            loss_streaks[current_length] += 1
        current_kind = None
        current_length = 0

    for outcome in outcomes:
        if outcome == 0:
            continue
        outcome_kind = "win" if outcome > 0 else "loss"
        if current_kind is None:
            current_kind = outcome_kind
            current_length = 1
        elif current_kind == outcome_kind:
            current_length += 1
        else:
            flush()
            current_kind = outcome_kind
            current_length = 1
    flush()

    signed = Counter({length: count for length, count in win_streaks.items()})
    signed.update({-length: count for length, count in loss_streaks.items()})
    return {
        "win_streaks": dict(sorted(win_streaks.items())),
        "loss_streaks": dict(sorted(loss_streaks.items())),
        "signed_streaks": dict(sorted(signed.items())),
    }


def theoretical_streak_probabilities(
    *,
    wins: int,
    losses: int,
    max_length: int,
) -> dict[str, dict[int, float]]:
    """Return geometric streak probability curves from observed non-push rates."""
    non_push = wins + losses
    if non_push == 0 or max_length <= 0:
        return {"win": {}, "loss": {}}
    p_win = wins / non_push
    p_loss = losses / non_push
    return {
        "win": {
            length: p_loss * (p_win ** (length - 1))
            for length in range(1, max_length + 1)
        },
        "loss": {
            length: p_win * (p_loss ** (length - 1))
            for length in range(1, max_length + 1)
        },
    }


def gaussian_smoothed_signed_streak_density(
    signed_streaks: dict[int, int],
    *,
    bandwidth: float = 1.0,
) -> dict[int, float]:
    """Return a simple Gaussian-smoothed density over signed integer streaks."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if not signed_streaks:
        return {}
    min_x = min(signed_streaks)
    max_x = max(signed_streaks)
    normalizer = bandwidth * sqrt(2.0 * pi)
    raw_density: dict[int, float] = {}
    total_count = sum(signed_streaks.values())
    for x_value in range(min_x, max_x + 1):
        if x_value == 0:
            continue
        density = 0.0
        for streak, count in signed_streaks.items():
            z_score = (x_value - streak) / bandwidth
            density += count * exp(-0.5 * z_score * z_score) / normalizer
        raw_density[x_value] = density / total_count
    return raw_density


def game_metrics_from_result(result: object) -> dict[str, Any]:
    """Build game-level experiment metrics from a SimulationResult-like object."""
    box = result.box_results[0]
    round_outcomes = [round_result.net_profit for round_result in result.round_results]
    round_wins = sum(1 for outcome in round_outcomes if outcome > 0)
    round_losses = sum(1 for outcome in round_outcomes if outcome < 0)
    round_pushes = sum(1 for outcome in round_outcomes if outcome == 0)
    rounds = result.rounds_played
    per_round_mean = sum(round_outcomes) / rounds if rounds else 0.0
    per_round_variance = (
        sum((outcome - per_round_mean) ** 2 for outcome in round_outcomes) / rounds
        if rounds
        else 0.0
    )
    streaks = streak_distributions(round_outcomes)
    max_streak_length = max(
        (abs(streak) for streak in streaks["signed_streaks"]),
        default=0,
    )

    return {
        "rounds": rounds,
        "initial_hands": rounds,
        "round_wins": round_wins,
        "round_losses": round_losses,
        "round_pushes": round_pushes,
        "hand_wins": box.wins,
        "hand_losses": box.losses,
        "hand_pushes": box.pushes,
        "player_blackjacks": box.blackjacks,
        "dealer_blackjacks": None,
        "blackjack_pushes": None,
        "splits": box.splits,
        "doubles": box.doubles,
        "busts": box.busts,
        "total_initial_wagered": result.initial_wagered,
        "total_action_wagered": result.action_wagered,
        "total_wagered": result.total_wagered,
        "net_profit": result.net_profit,
        "win_rate_per_initial_round": _rate(round_wins, rounds),
        "loss_rate_per_initial_round": _rate(round_losses, rounds),
        "push_rate_per_initial_round": _rate(round_pushes, rounds),
        "player_blackjack_rate_per_initial_hand": _rate(box.blackjacks, rounds),
        "expected_iid_player_blackjack_rate": 8.0 / 169.0,
        "split_rate_per_initial_hand": _rate(box.splits, rounds),
        "double_rate_per_initial_hand": _rate(box.doubles, rounds),
        "bust_rate_per_resolved_hand": _rate(box.busts, box.hands_resolved),
        "edge_per_initial_wager": result.edge_per_initial_wager,
        "edge_per_total_wager": result.edge_per_total_wager,
        "average_profit_per_round": result.average_profit_per_round,
        "per_round_profit_stddev": sqrt(per_round_variance),
        "cumulative_profit_path": [
            round_result.cumulative_profit_after_round
            for round_result in result.round_results
        ],
        "streaks": streaks,
        "theoretical_streak_probabilities": theoretical_streak_probabilities(
            wins=round_wins,
            losses=round_losses,
            max_length=max_streak_length,
        ),
    }


def _rate(count: int | float, denominator: int | float) -> float:
    return count / denominator if denominator else 0.0
