# SPDX-License-Identifier: GPL-3.0-or-later

"""Counterfactual next-round values for one through seven active boxes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

from experiments.plots import plot_counterfactual_heatmap
from experiments.single_box_game_validation import student_t_summary
from shufflemaster_sim.card_sources import (
    One2SixCardSource,
    One2SixConfig,
    PhysicalIidCardSource,
)
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
    CasinoBlackjackStrategy,
)
from shufflemaster_sim.hand_values import (
    is_bust,
    is_natural_blackjack,
    split_value_from_rank,
)
from shufflemaster_sim.state import TableState
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)

SourceName = Literal["physical_iid", "one2six"]
StateBucket = Literal[
    "strong_high_heavy",
    "moderate_high_heavy",
    "neutral",
    "moderate_low_heavy",
    "strong_low_heavy",
]
CardCategory = Literal["low", "neutral", "high"]
DealerUpcardCategory = Literal["ace", "ten_value", "low", "neutral"]
AggregateOutcome = Literal["win", "loss", "push"]

SOURCE_NAMES: Final[tuple[SourceName, ...]] = ("physical_iid", "one2six")
STATE_BUCKETS: Final[tuple[StateBucket, ...]] = (
    "strong_high_heavy",
    "moderate_high_heavy",
    "neutral",
    "moderate_low_heavy",
    "strong_low_heavy",
)
LOW_RANKS: Final[frozenset[Rank]] = frozenset({"2", "3", "4", "5", "6"})
NEUTRAL_RANKS: Final[frozenset[Rank]] = frozenset({"7", "8", "9"})
TEN_VALUE_RANKS: Final[frozenset[Rank]] = frozenset({"T", "J", "Q", "K"})
SPARSE_STATE_THRESHOLD: Final[int] = 100
COUNTERFACTUAL_ACTION_FIELDS: Final[tuple[str, ...]] = (
    "source",
    "seed",
    "state_id",
    "rack_size",
    "rack_hi_lo_count",
    "rack_low_count",
    "rack_neutral_count",
    "rack_ten_value_count",
    "rack_ace_count",
    "normalized_exclusion_score",
    "state_bucket",
    "box_count",
    "initial_wager",
    "additional_wager",
    "total_wager",
    "total_player_net",
    "aggregate_outcome",
    "cards_consumed",
    "dealer_upcard_category",
    "player_blackjacks",
    "double_actions",
    "split_actions",
)


@dataclass(frozen=True, slots=True)
class MultiBoxCounterfactualConfig:
    """Configuration for common-state next-round box-count branches."""

    states_per_seed: int = 2_000
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    burn_in_rounds: int = 1_000
    sample_interval_rounds: int = 5
    base_bet: float = 10.0
    deck_count: int = 6
    box_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    output_dir: Path = Path("experiments/outputs/multi_box_counterfactual_5x2000")

    def __post_init__(self) -> None:
        if self.states_per_seed <= 0:
            raise ValueError("states_per_seed must be positive.")
        if self.burn_in_rounds < 0:
            raise ValueError("burn_in_rounds must be non-negative.")
        if self.sample_interval_rounds <= 0:
            raise ValueError("sample_interval_rounds must be positive.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        if self.deck_count != 6:
            raise ValueError("counterfactual comparison requires exactly six decks.")
        if not self.seeds:
            raise ValueError("at least one seed must be supplied.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique independent run identifiers.")
        if not self.box_counts:
            raise ValueError("at least one box count must be supplied.")
        if len(set(self.box_counts)) != len(self.box_counts):
            raise ValueError("box_counts must be unique.")
        if any(
            type(count) is not int or not 1 <= count <= 7 for count in self.box_counts
        ):
            raise ValueError("box_counts must be unique integers from 1 through 7.")


@dataclass(frozen=True, slots=True)
class ObservableRackState:
    """Player-observable discard-rack composition at the betting boundary."""

    rack_size: int
    rack_hi_lo_count: int
    rack_low_count: int
    rack_neutral_count: int
    rack_ten_value_count: int
    rack_ace_count: int
    normalized_exclusion_score: float
    state_bucket: StateBucket

    def as_feature_record(self) -> dict[str, int | float | str]:
        """Return only the allowed player-observable feature fields."""
        return asdict(self)


CardSourceType = PhysicalIidCardSource | One2SixCardSource


@dataclass(frozen=True, slots=True)
class CounterfactualSnapshot:
    """Complete hidden branch state, retained only inside the experiment."""

    game: CasinoBlackjackGame
    card_source: CardSourceType
    next_round_index: int


@dataclass(frozen=True, slots=True)
class SampledState:
    """Observable identity and features for one sampled betting boundary."""

    source: SourceName
    seed: int
    state_id: str
    observable: ObservableRackState


@dataclass(frozen=True, slots=True)
class BoxPositionResult:
    """One box position's result inside a counterfactual branch."""

    box_position: int
    initial_wager: float
    net_player_result: float
    player_blackjacks: int
    double_actions: int
    split_actions: int
    first_card_category: CardCategory
    second_card_category: CardCategory


@dataclass(frozen=True, slots=True)
class BranchResult:
    """One next-round outcome for a sampled state and box-count action."""

    state: SampledState
    box_count: int
    initial_wager: float
    additional_wager: float
    total_wager: float
    total_player_net: float
    aggregate_outcome: AggregateOutcome
    cards_consumed: int
    dealer_upcard_category: DealerUpcardCategory
    dealer_blackjack_observed: bool
    dealer_blackjack: bool
    dealer_completion_observed: bool
    dealer_bust: bool
    player_blackjacks: int
    double_actions: int
    split_actions: int
    box_positions: tuple[BoxPositionResult, ...]

    def as_action_record(self) -> dict[str, Any]:
        """Return the compact public row without hidden source state."""
        record: dict[str, Any] = {
            "source": self.state.source,
            "seed": self.state.seed,
            "state_id": self.state.state_id,
            **self.state.observable.as_feature_record(),
            "box_count": self.box_count,
            "initial_wager": self.initial_wager,
            "additional_wager": self.additional_wager,
            "total_wager": self.total_wager,
            "total_player_net": self.total_player_net,
            "aggregate_outcome": self.aggregate_outcome,
            "cards_consumed": self.cards_consumed,
            "dealer_upcard_category": self.dealer_upcard_category,
            "player_blackjacks": self.player_blackjacks,
            "double_actions": self.double_actions,
            "split_actions": self.split_actions,
        }
        if tuple(record) != COUNTERFACTUAL_ACTION_FIELDS:
            raise RuntimeError(
                "Counterfactual action export fields changed unexpectedly."
            )
        return record


@dataclass(frozen=True, slots=True)
class MarginalRecord:
    """Paired next-round result from adding one box to the same state."""

    source: SourceName
    seed: int
    state_id: str
    state_bucket: StateBucket
    added_box_number: int
    marginal_net: float


def observable_rack_state(
    cards: Sequence[Card], *, deck_count: int = 6
) -> ObservableRackState:
    """Calculate the fixed observable rack feature set."""
    if deck_count != 6:
        raise ValueError("observable exclusion score currently requires six decks.")
    rack_size = len(cards)
    rack_low_count = sum(card.rank in LOW_RANKS for card in cards)
    rack_neutral_count = sum(card.rank in NEUTRAL_RANKS for card in cards)
    rack_ten_value_count = sum(card.rank in TEN_VALUE_RANKS for card in cards)
    rack_ace_count = sum(card.rank == "A" for card in cards)
    if (
        rack_low_count + rack_neutral_count + rack_ten_value_count + rack_ace_count
        != rack_size
    ):
        raise RuntimeError("Observable rack category counts do not reconcile.")
    rack_hi_lo_count = rack_low_count - rack_ten_value_count - rack_ace_count
    cards_inside_source = deck_count * 52 - rack_size
    if cards_inside_source <= 0:
        raise ValueError("visible rack leaves no cards inside the six-deck source.")
    return ObservableRackState(
        rack_size=rack_size,
        rack_hi_lo_count=rack_hi_lo_count,
        rack_low_count=rack_low_count,
        rack_neutral_count=rack_neutral_count,
        rack_ten_value_count=rack_ten_value_count,
        rack_ace_count=rack_ace_count,
        normalized_exclusion_score=rack_hi_lo_count / cards_inside_source,
        state_bucket=state_bucket(rack_hi_lo_count),
    )


def state_bucket(rack_hi_lo_count: int) -> StateBucket:
    """Return the predefined observable Hi-Lo bucket."""
    if rack_hi_lo_count <= -6:
        return "strong_high_heavy"
    if rack_hi_lo_count <= -3:
        return "moderate_high_heavy"
    if rack_hi_lo_count <= 2:
        return "neutral"
    if rack_hi_lo_count <= 5:
        return "moderate_low_heavy"
    return "strong_low_heavy"


def clone_snapshot(snapshot: CounterfactualSnapshot) -> CounterfactualSnapshot:
    """Deep-copy complete game, source, RNG, buffer, shelf, and rack state."""
    return deepcopy(snapshot)


def branch_from_snapshot(
    snapshot: CounterfactualSnapshot,
    *,
    sampled_state: SampledState,
    box_count: int,
    base_bet: float,
    strategy: CasinoBlackjackStrategy,
) -> BranchResult:
    """Play one isolated box-count branch from a complete common state."""
    if not 1 <= box_count <= 7:
        raise ValueError("box_count must be between 1 and 7.")
    branch = clone_snapshot(snapshot)
    branch.game.config = replace(
        branch.game.config,
        base_bet=base_bet,
        box_count=box_count,
        box_bets={box_id: base_bet for box_id in range(1, box_count + 1)},
    )
    branch_strategy = deepcopy(strategy)
    draw_count_before = branch.card_source.draw_count
    table, initial_cards = _play_round_and_capture_initial_cards(
        game=branch.game,
        card_source=branch.card_source,
        strategy=branch_strategy,
        round_index=branch.next_round_index,
    )
    cards_consumed = branch.card_source.draw_count - draw_count_before
    _assert_source_invariants(branch.card_source, branch.game.pending_discard_rack)
    return _branch_result(
        state=sampled_state,
        box_count=box_count,
        base_bet=base_bet,
        table=table,
        initial_cards=initial_cards,
        cards_consumed=cards_consumed,
    )


def paired_marginal_records(
    branches: Sequence[BranchResult],
    *,
    added_box_numbers: Sequence[int],
) -> list[MarginalRecord]:
    """Build paired marginal values, requiring matching state IDs."""
    by_state: dict[str, dict[int, BranchResult]] = defaultdict(dict)
    for branch in branches:
        current = by_state[branch.state.state_id]
        if branch.box_count in current:
            raise RuntimeError("Duplicate box-count branch for sampled state.")
        current[branch.box_count] = branch

    marginal_records: list[MarginalRecord] = []
    for state_id, state_branches in by_state.items():
        for added_box in added_box_numbers:
            lower = state_branches.get(added_box - 1)
            upper = state_branches.get(added_box)
            if lower is None or upper is None:
                raise RuntimeError(
                    f"Missing paired branches for state {state_id!r}, box {added_box}."
                )
            if lower.state.state_id != upper.state.state_id:
                raise RuntimeError("Marginal branches do not share a sampled state.")
            marginal_records.append(
                MarginalRecord(
                    source=upper.state.source,
                    seed=upper.state.seed,
                    state_id=state_id,
                    state_bucket=upper.state.observable.state_bucket,
                    added_box_number=added_box,
                    marginal_net=upper.total_player_net - lower.total_player_net,
                )
            )
    return marginal_records


def run_multi_box_counterfactual_experiment(
    config: MultiBoxCounterfactualConfig,
    *,
    strategy: CasinoBlackjackStrategy | None = None,
) -> dict[str, Any]:
    """Generate common states, branch box-count actions, and write reports."""
    fixed_strategy = strategy or PublishedApproxCasinoStrategy()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sampled_states: list[SampledState] = []
    branch_results: list[BranchResult] = []
    added_boxes = tuple(
        box_count
        for box_count in config.box_counts
        if box_count >= 2 and box_count - 1 in config.box_counts
    )

    action_path = config.output_dir / "counterfactual_actions.csv"
    with action_path.open("w", encoding="utf-8", newline="") as action_file:
        writer = csv.DictWriter(action_file, fieldnames=COUNTERFACTUAL_ACTION_FIELDS)
        writer.writeheader()
        for source_name in SOURCE_NAMES:
            for seed in config.seeds:
                states, branches = _run_source_seed(
                    config=config,
                    source_name=source_name,
                    seed=seed,
                    strategy=fixed_strategy,
                )
                sampled_states.extend(states)
                branch_results.extend(branches)
                writer.writerows(branch.as_action_record() for branch in branches)

    expected_states = len(SOURCE_NAMES) * len(config.seeds) * config.states_per_seed
    expected_branches = expected_states * len(config.box_counts)
    if len(sampled_states) != expected_states:
        raise RuntimeError("Sampled state count does not reconcile with configuration.")
    if len(branch_results) != expected_branches:
        raise RuntimeError("Branch rows do not reconcile with states and box counts.")

    marginal_records = paired_marginal_records(
        branch_results,
        added_box_numbers=added_boxes,
    )
    state_frequency_rows = _state_frequency_rows(config, sampled_states)
    per_seed_action_rows, aggregate_action_rows = _state_action_rows(
        config,
        branch_results,
    )
    marginal_rows = _marginal_rows(config, marginal_records, added_boxes)
    position_rows = _box_position_rows(config, branch_results)
    _validate_output_reconciliation(
        config=config,
        sampled_states=sampled_states,
        branches=branch_results,
        marginal_records=marginal_records,
        position_rows=position_rows,
        added_boxes=added_boxes,
    )

    plot_paths = _write_heatmaps(
        config.output_dir,
        aggregate_action_rows,
        marginal_rows,
        box_counts=config.box_counts,
        added_boxes=added_boxes,
    )
    signal_checks = _signal_checks(
        aggregate_action_rows,
        state_frequency_rows,
        marginal_rows,
    )
    summary = {
        "experiment": "multi_box_counterfactual_action_value",
        "purpose": (
            "Counterfactual next-round action values from player-observable "
            "discard-rack composition; no policy is selected or validated."
        ),
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "behavior_policy_boxes": 1,
            "flat_betting": True,
            "strategy_implementation": type(fixed_strategy).__name__,
            "actual_rounds_before_first_sample": max(1, config.burn_in_rounds),
        },
        "observable_feature_fields": list(ObservableRackState.__dataclass_fields__),
        "hidden_state_exported": False,
        "state_frequency": state_frequency_rows,
        "state_action_values": aggregate_action_rows,
        "per_seed_state_action_values": per_seed_action_rows,
        "marginal_box_values": marginal_rows,
        "box_position_summary": position_rows,
        "signal_checks": signal_checks,
        "plot_paths": plot_paths,
        "definitions": {
            "state_timing": (
                "After the preceding round is settled and before its visible rack "
                "is returned during the next initial deal."
            ),
            "normalized_exclusion_score": ("rack_hi_lo_count / (312 - rack_size)"),
            "marginal_net": "paired B-box net minus (B-1)-box net for one state",
            "uncertainty_unit": "independent seed-level estimates",
            "sparse_threshold": SPARSE_STATE_THRESHOLD,
        },
    }

    _write_json(config.output_dir / "summary.json", summary)
    _write_csv(config.output_dir / "state_frequency.csv", state_frequency_rows)
    _write_csv(config.output_dir / "state_action_values.csv", aggregate_action_rows)
    _write_csv(
        config.output_dir / "per_seed_state_action_values.csv",
        per_seed_action_rows,
    )
    _write_csv(config.output_dir / "marginal_box_values.csv", marginal_rows)
    _write_csv(config.output_dir / "box_position_summary.csv", position_rows)
    (config.output_dir / "summary.md").write_text(
        _summary_markdown(
            config,
            state_frequency_rows,
            aggregate_action_rows,
            marginal_rows,
            signal_checks,
            plot_paths,
        ),
        encoding="utf-8",
    )
    return summary


def _run_source_seed(
    *,
    config: MultiBoxCounterfactualConfig,
    source_name: SourceName,
    seed: int,
    strategy: CasinoBlackjackStrategy,
) -> tuple[list[SampledState], list[BranchResult]]:
    source = _make_source(source_name, config.deck_count, seed)
    behavior_game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            base_bet=config.base_bet,
            box_count=1,
            box_bets={1: config.base_bet},
            deck_count=config.deck_count,
        )
    )
    next_round_index = 0
    for _ in range(max(1, config.burn_in_rounds)):
        behavior_game.play_round(
            round_index=next_round_index,
            card_source=source,
            strategy=strategy,
        )
        next_round_index += 1
    _assert_source_invariants(source, behavior_game.pending_discard_rack)

    states: list[SampledState] = []
    branches: list[BranchResult] = []
    for state_index in range(config.states_per_seed):
        observable = observable_rack_state(
            behavior_game.pending_discard_rack,
            deck_count=config.deck_count,
        )
        sampled_state = SampledState(
            source=source_name,
            seed=seed,
            state_id=f"{source_name}:{seed}:{state_index}",
            observable=observable,
        )
        snapshot = CounterfactualSnapshot(
            game=deepcopy(behavior_game),
            card_source=deepcopy(source),
            next_round_index=next_round_index,
        )
        snapshot_draw_count = snapshot.card_source.draw_count
        snapshot_rack = snapshot.game.pending_discard_rack
        state_branches = [
            branch_from_snapshot(
                snapshot,
                sampled_state=sampled_state,
                box_count=box_count,
                base_bet=config.base_bet,
                strategy=strategy,
            )
            for box_count in config.box_counts
        ]
        if snapshot.card_source.draw_count != snapshot_draw_count:
            raise RuntimeError("Counterfactual branch mutated its original snapshot.")
        if snapshot.game.pending_discard_rack != snapshot_rack:
            raise RuntimeError("Counterfactual branch mutated the snapshot rack.")
        states.append(sampled_state)
        branches.extend(state_branches)

        if state_index + 1 < config.states_per_seed:
            for _ in range(config.sample_interval_rounds):
                behavior_game.play_round(
                    round_index=next_round_index,
                    card_source=source,
                    strategy=strategy,
                )
                next_round_index += 1
            _assert_source_invariants(source, behavior_game.pending_discard_rack)
    return states, branches


def _make_source(source_name: SourceName, deck_count: int, seed: int) -> CardSourceType:
    if source_name == "physical_iid":
        return PhysicalIidCardSource(deck_count=deck_count, seed=seed)
    return One2SixCardSource(
        config=One2SixConfig(
            deck_count=deck_count,
            strict_invariants=False,
            retain_event_telemetry=False,
            retain_ejection_records=False,
            retain_accepted_discard_history=False,
        ),
        seed=seed,
    )


def _play_round_and_capture_initial_cards(
    *,
    game: CasinoBlackjackGame,
    card_source: CardSourceType,
    strategy: CasinoBlackjackStrategy,
    round_index: int,
) -> tuple[TableState, dict[int, tuple[Card, Card]]]:
    card_source.before_round()
    game.burn_initial_card(card_source)
    table = game.create_table(round_index)
    game.deal_initial_cards(table, card_source)
    initial_cards = {
        box.box_id: (box.hands[0].cards[0], box.hands[0].cards[1])
        for box in table.boxes
    }
    game.settle_immediate_blackjacks(table)
    game.play_player_hands(table, card_source, strategy)
    game.play_dealer(table, card_source)
    game.settle(table)
    game.collect_remaining_layout_cards(table)
    game.stage_discard_rack_for_next_round(table)
    return table, initial_cards


def _branch_result(
    *,
    state: SampledState,
    box_count: int,
    base_bet: float,
    table: TableState,
    initial_cards: Mapping[int, tuple[Card, Card]],
    cards_consumed: int,
) -> BranchResult:
    box_positions: list[BoxPositionResult] = []
    for box in table.boxes:
        cards = initial_cards[box.box_id]
        box_positions.append(
            BoxPositionResult(
                box_position=box.box_id,
                initial_wager=box.base_bet,
                net_player_result=sum(hand.net_result for hand in box.hands),
                player_blackjacks=sum(
                    is_natural_blackjack(
                        hand.cards,
                        blackjack_eligible=hand.blackjack_eligible,
                    )
                    for hand in box.hands
                ),
                double_actions=sum(hand.is_doubled for hand in box.hands),
                split_actions=int(len(box.hands) > 1),
                first_card_category=card_category(cards[0].rank),
                second_card_category=card_category(cards[1].rank),
            )
        )

    initial_wager = box_count * base_bet
    total_wager = sum(
        hand.wager * (2.0 if hand.is_doubled else 1.0)
        for box in table.boxes
        for hand in box.hands
    )
    total_player_net = sum(position.net_player_result for position in box_positions)
    table_net = sum(hand.net_result for box in table.boxes for hand in box.hands)
    if total_player_net != table_net:
        raise RuntimeError("Branch total does not equal the sum of active boxes.")
    if len(box_positions) != box_count:
        raise RuntimeError("Branch did not create the requested number of boxes.")
    dealer_upcard = table.dealer.cards[0]
    dealer_can_have_blackjack = split_value_from_rank(dealer_upcard.rank) in {"A", 10}
    dealer_blackjack_observed = (
        len(table.dealer.cards) >= 2 or not dealer_can_have_blackjack
    )
    dealer_completion_observed = len(table.dealer.cards) >= 2
    return BranchResult(
        state=state,
        box_count=box_count,
        initial_wager=initial_wager,
        additional_wager=total_wager - initial_wager,
        total_wager=total_wager,
        total_player_net=total_player_net,
        aggregate_outcome=_aggregate_outcome(total_player_net),
        cards_consumed=cards_consumed,
        dealer_upcard_category=dealer_upcard_category(dealer_upcard.rank),
        dealer_blackjack_observed=dealer_blackjack_observed,
        dealer_blackjack=is_natural_blackjack(table.dealer.cards),
        dealer_completion_observed=dealer_completion_observed,
        dealer_bust=dealer_completion_observed and is_bust(table.dealer.cards),
        player_blackjacks=sum(item.player_blackjacks for item in box_positions),
        double_actions=sum(item.double_actions for item in box_positions),
        split_actions=sum(item.split_actions for item in box_positions),
        box_positions=tuple(box_positions),
    )


def card_category(rank: Rank) -> CardCategory:
    """Return the three-level Hi-Lo card category."""
    if rank in LOW_RANKS:
        return "low"
    if rank in NEUTRAL_RANKS:
        return "neutral"
    return "high"


def dealer_upcard_category(rank: Rank) -> DealerUpcardCategory:
    """Return the dealer category retained by the compact action output."""
    if rank == "A":
        return "ace"
    if rank in TEN_VALUE_RANKS:
        return "ten_value"
    if rank in LOW_RANKS:
        return "low"
    return "neutral"


def _aggregate_outcome(net_result: float) -> AggregateOutcome:
    if net_result > 0:
        return "win"
    if net_result < 0:
        return "loss"
    return "push"


def _assert_source_invariants(
    source: CardSourceType,
    pending_rack: Sequence[Card],
) -> None:
    if isinstance(source, One2SixCardSource):
        source.assert_invariants(pending_rack)


def _state_frequency_rows(
    config: MultiBoxCounterfactualConfig,
    states: Sequence[SampledState],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        for seed_or_aggregate in (*config.seeds, "aggregate"):
            relevant = [
                state
                for state in states
                if state.source == source_name
                and (
                    seed_or_aggregate == "aggregate" or state.seed == seed_or_aggregate
                )
            ]
            denominator = len(relevant)
            for bucket in STATE_BUCKETS:
                bucket_states = [
                    state
                    for state in relevant
                    if state.observable.state_bucket == bucket
                ]
                count = len(bucket_states)
                rows.append(
                    {
                        "source": source_name,
                        "seed_or_aggregate": seed_or_aggregate,
                        "state_bucket": bucket,
                        "state_count": count,
                        "state_proportion": _rate(count, denominator),
                        "mean_rack_size": _mean(
                            [state.observable.rack_size for state in bucket_states]
                        ),
                        "mean_rack_hi_lo_count": _mean(
                            [
                                state.observable.rack_hi_lo_count
                                for state in bucket_states
                            ]
                        ),
                        "sparse": count < SPARSE_STATE_THRESHOLD,
                    }
                )
    return rows


def _state_action_rows(
    config: MultiBoxCounterfactualConfig,
    branches: Sequence[BranchResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        for bucket in STATE_BUCKETS:
            for box_count in config.box_counts:
                seed_rows: list[dict[str, Any]] = []
                for seed in config.seeds:
                    group = [
                        branch
                        for branch in branches
                        if branch.state.source == source_name
                        and branch.state.seed == seed
                        and branch.state.observable.state_bucket == bucket
                        and branch.box_count == box_count
                    ]
                    row = _summarize_action_group(
                        group,
                        source=source_name,
                        state_bucket_name=bucket,
                        box_count=box_count,
                    )
                    row["seed"] = seed
                    per_seed.append(row)
                    if group:
                        seed_rows.append(row)

                group = [
                    branch
                    for branch in branches
                    if branch.state.source == source_name
                    and branch.state.observable.state_bucket == bucket
                    and branch.box_count == box_count
                ]
                row = _summarize_action_group(
                    group,
                    source=source_name,
                    state_bucket_name=bucket,
                    box_count=box_count,
                )
                uncertainty = student_t_summary(
                    [item["edge_per_initial_wager"] for item in seed_rows]
                )
                row.update(_uncertainty_fields(uncertainty, prefix="seed_edge"))
                row["contributing_seeds"] = len(seed_rows)
                aggregate.append(row)
    return per_seed, aggregate


def _summarize_action_group(
    group: Sequence[BranchResult],
    *,
    source: SourceName,
    state_bucket_name: StateBucket,
    box_count: int,
) -> dict[str, Any]:
    sampled_states = len(group)
    initial_wager = sum(branch.initial_wager for branch in group)
    additional_wager = sum(branch.additional_wager for branch in group)
    total_wager = sum(branch.total_wager for branch in group)
    total_net = sum(branch.total_player_net for branch in group)
    active_box_rounds = sampled_states * box_count
    dealer_blackjack_observations = sum(
        branch.dealer_blackjack_observed for branch in group
    )
    dealer_completion_observations = sum(
        branch.dealer_completion_observed for branch in group
    )
    row = {
        "source": source,
        "state_bucket": state_bucket_name,
        "box_count": box_count,
        "sampled_states": sampled_states,
        "total_initial_wager": initial_wager,
        "total_additional_action_wager": additional_wager,
        "total_wager": total_wager,
        "total_player_net": total_net,
        "mean_total_player_net": _rate_or_none(total_net, sampled_states),
        "edge_per_initial_wager": _rate_or_none(total_net, initial_wager),
        "edge_per_total_wager": _rate_or_none(total_net, total_wager),
        "positive_round_rate": _rate_or_none(
            sum(branch.aggregate_outcome == "win" for branch in group),
            sampled_states,
        ),
        "negative_round_rate": _rate_or_none(
            sum(branch.aggregate_outcome == "loss" for branch in group),
            sampled_states,
        ),
        "zero_round_rate": _rate_or_none(
            sum(branch.aggregate_outcome == "push" for branch in group),
            sampled_states,
        ),
        "player_blackjacks": sum(branch.player_blackjacks for branch in group),
        "blackjack_rate_per_active_box": _rate_or_none(
            sum(branch.player_blackjacks for branch in group), active_box_rounds
        ),
        "double_actions": sum(branch.double_actions for branch in group),
        "double_rate_per_active_box": _rate_or_none(
            sum(branch.double_actions for branch in group), active_box_rounds
        ),
        "split_actions": sum(branch.split_actions for branch in group),
        "split_rate_per_active_box": _rate_or_none(
            sum(branch.split_actions for branch in group), active_box_rounds
        ),
        "dealer_upcard_ace_rate": _category_rate(group, "ace"),
        "dealer_upcard_ten_value_rate": _category_rate(group, "ten_value"),
        "dealer_upcard_low_rate": _category_rate(group, "low"),
        "dealer_upcard_neutral_rate": _category_rate(group, "neutral"),
        "dealer_blackjack_observations": dealer_blackjack_observations,
        "dealer_blackjack_rate_observed": _rate_or_none(
            sum(branch.dealer_blackjack for branch in group),
            dealer_blackjack_observations,
        ),
        "dealer_completion_observations": dealer_completion_observations,
        "dealer_bust_rate_observed": _rate_or_none(
            sum(branch.dealer_bust for branch in group),
            dealer_completion_observations,
        ),
        "mean_cards_consumed": _mean([branch.cards_consumed for branch in group]),
        "sparse": sampled_states < SPARSE_STATE_THRESHOLD,
    }
    expected_initial = (
        sampled_states
        * box_count
        * (group[0].box_positions[0].initial_wager if group else 0.0)
    )
    if initial_wager != expected_initial:
        raise RuntimeError("Aggregate initial wager does not reconcile.")
    return row


def _marginal_rows(
    config: MultiBoxCounterfactualConfig,
    marginals: Sequence[MarginalRecord],
    added_boxes: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        for bucket in STATE_BUCKETS:
            for added_box in added_boxes:
                seed_rows: list[dict[str, Any]] = []
                for seed in config.seeds:
                    group = [
                        item
                        for item in marginals
                        if item.source == source_name
                        and item.seed == seed
                        and item.state_bucket == bucket
                        and item.added_box_number == added_box
                    ]
                    row = _summarize_marginal_group(
                        group,
                        source=source_name,
                        state_bucket_name=bucket,
                        added_box=added_box,
                        base_bet=config.base_bet,
                    )
                    row["seed_or_aggregate"] = seed
                    rows.append(row)
                    if group:
                        seed_rows.append(row)

                group = [
                    item
                    for item in marginals
                    if item.source == source_name
                    and item.state_bucket == bucket
                    and item.added_box_number == added_box
                ]
                row = _summarize_marginal_group(
                    group,
                    source=source_name,
                    state_bucket_name=bucket,
                    added_box=added_box,
                    base_bet=config.base_bet,
                )
                uncertainty = student_t_summary(
                    [item["mean_marginal_net"] for item in seed_rows]
                )
                row.update(_uncertainty_fields(uncertainty, prefix="seed_marginal_net"))
                row["contributing_seeds"] = len(seed_rows)
                row["seed_or_aggregate"] = "aggregate"
                rows.append(row)
    return rows


def _summarize_marginal_group(
    group: Sequence[MarginalRecord],
    *,
    source: SourceName,
    state_bucket_name: StateBucket,
    added_box: int,
    base_bet: float,
) -> dict[str, Any]:
    count = len(group)
    mean_marginal = _mean([item.marginal_net for item in group])
    return {
        "source": source,
        "state_bucket": state_bucket_name,
        "added_box_number": added_box,
        "paired_state_count": count,
        "mean_marginal_net": mean_marginal,
        "marginal_return_per_added_box": (
            mean_marginal / base_bet if mean_marginal is not None else None
        ),
        "proportion_improved": _rate_or_none(
            sum(item.marginal_net > 0 for item in group), count
        ),
        "sparse": count < SPARSE_STATE_THRESHOLD,
    }


def _box_position_rows(
    config: MultiBoxCounterfactualConfig,
    branches: Sequence[BranchResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        for bucket in STATE_BUCKETS:
            for branch_box_count in config.box_counts:
                branch_group = [
                    branch
                    for branch in branches
                    if branch.state.source == source_name
                    and branch.state.observable.state_bucket == bucket
                    and branch.box_count == branch_box_count
                ]
                for box_position in range(1, branch_box_count + 1):
                    positions = [
                        branch.box_positions[box_position - 1]
                        for branch in branch_group
                    ]
                    rounds = len(positions)
                    initial_wager = sum(item.initial_wager for item in positions)
                    net = sum(item.net_player_result for item in positions)
                    rows.append(
                        {
                            "source": source_name,
                            "state_bucket": bucket,
                            "branch_box_count": branch_box_count,
                            "box_position": box_position,
                            "rounds_played": rounds,
                            "initial_wager": initial_wager,
                            "net_player_result": net,
                            "edge_per_initial_wager": _rate_or_none(net, initial_wager),
                            "blackjack_rate": _rate_or_none(
                                sum(item.player_blackjacks for item in positions),
                                rounds,
                            ),
                            "double_rate": _rate_or_none(
                                sum(item.double_actions for item in positions), rounds
                            ),
                            "split_rate": _rate_or_none(
                                sum(item.split_actions for item in positions), rounds
                            ),
                            "first_card_low_rate": _position_category_rate(
                                positions, "first_card_category", "low"
                            ),
                            "first_card_neutral_rate": _position_category_rate(
                                positions, "first_card_category", "neutral"
                            ),
                            "first_card_high_rate": _position_category_rate(
                                positions, "first_card_category", "high"
                            ),
                            "second_card_low_rate": _position_category_rate(
                                positions, "second_card_category", "low"
                            ),
                            "second_card_neutral_rate": _position_category_rate(
                                positions, "second_card_category", "neutral"
                            ),
                            "second_card_high_rate": _position_category_rate(
                                positions, "second_card_category", "high"
                            ),
                            "sparse": rounds < SPARSE_STATE_THRESHOLD,
                        }
                    )
    return rows


def _validate_output_reconciliation(
    *,
    config: MultiBoxCounterfactualConfig,
    sampled_states: Sequence[SampledState],
    branches: Sequence[BranchResult],
    marginal_records: Sequence[MarginalRecord],
    position_rows: Sequence[Mapping[str, Any]],
    added_boxes: Sequence[int],
) -> None:
    state_ids = {state.state_id for state in sampled_states}
    if len(state_ids) != len(sampled_states):
        raise RuntimeError("Sampled state IDs are not unique.")
    if {branch.state.state_id for branch in branches} != state_ids:
        raise RuntimeError("Branch state IDs do not reconcile with sampled states.")
    expected_branch_count = len(sampled_states) * len(config.box_counts)
    if len(branches) != expected_branch_count:
        raise RuntimeError("Branch count does not equal states times actions.")
    expected_marginals = len(sampled_states) * len(added_boxes)
    if len(marginal_records) != expected_marginals:
        raise RuntimeError("Marginal rows do not reconcile with paired branches.")

    for source_name in SOURCE_NAMES:
        for bucket in STATE_BUCKETS:
            for box_count in config.box_counts:
                group = [
                    branch
                    for branch in branches
                    if branch.state.source == source_name
                    and branch.state.observable.state_bucket == bucket
                    and branch.box_count == box_count
                ]
                rows = [
                    row
                    for row in position_rows
                    if row["source"] == source_name
                    and row["state_bucket"] == bucket
                    and row["branch_box_count"] == box_count
                ]
                if sum(row["initial_wager"] for row in rows) != sum(
                    branch.initial_wager for branch in group
                ):
                    raise RuntimeError(
                        "Position wagers do not reconcile with branches."
                    )
                if sum(row["net_player_result"] for row in rows) != sum(
                    branch.total_player_net for branch in group
                ):
                    raise RuntimeError("Position net does not reconcile with branches.")


def _write_heatmaps(
    output_dir: Path,
    action_rows: Sequence[Mapping[str, Any]],
    marginal_rows: Sequence[Mapping[str, Any]],
    *,
    box_counts: Sequence[int],
    added_boxes: Sequence[int],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    aggregate_marginals = [
        row for row in marginal_rows if row["seed_or_aggregate"] == "aggregate"
    ]
    for source_name in SOURCE_NAMES:
        edge_name = f"state_action_edge_heatmap_{source_name}.png"
        plot_counterfactual_heatmap(
            [row for row in action_rows if row["source"] == source_name],
            row_keys=STATE_BUCKETS,
            column_keys=box_counts,
            row_field="state_bucket",
            column_field="box_count",
            value_field="edge_per_initial_wager",
            sparse_field="sparse",
            title=f"{_source_label(source_name)} state-conditioned edge",
            x_label="Active boxes",
            y_label="Visible-rack state bucket",
            output_path=output_dir / edge_name,
            percentage=True,
        )
        paths[f"state_action_edge_{source_name}"] = edge_name

        marginal_name = f"marginal_box_value_heatmap_{source_name}.png"
        plot_counterfactual_heatmap(
            [row for row in aggregate_marginals if row["source"] == source_name],
            row_keys=STATE_BUCKETS,
            column_keys=added_boxes,
            row_field="state_bucket",
            column_field="added_box_number",
            value_field="marginal_return_per_added_box",
            sparse_field="sparse",
            title=f"{_source_label(source_name)} paired marginal box return",
            x_label="Added box number",
            y_label="Visible-rack state bucket",
            output_path=output_dir / marginal_name,
            percentage=True,
        )
        paths[f"marginal_box_value_{source_name}"] = marginal_name
    return paths


def _signal_checks(
    action_rows: Sequence[Mapping[str, Any]],
    frequency_rows: Sequence[Mapping[str, Any]],
    marginal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for source_name in SOURCE_NAMES:
        source_rows = [row for row in action_rows if row["source"] == source_name]
        ordered_by_box: dict[str, bool] = {}
        low_heavy_vs_neutral: dict[str, bool | None] = {}
        for box_count in sorted({int(row["box_count"]) for row in source_rows}):
            rows = {
                row["state_bucket"]: row
                for row in source_rows
                if row["box_count"] == box_count
            }
            edges = [
                rows[bucket]["edge_per_initial_wager"]
                for bucket in STATE_BUCKETS
                if not rows[bucket]["sparse"]
            ]
            ordered_by_box[str(box_count)] = all(
                left is not None and right is not None and left <= right
                for left, right in zip(edges, edges[1:], strict=False)
            )
            low_values = [
                rows[bucket]["edge_per_initial_wager"]
                for bucket in ("moderate_low_heavy", "strong_low_heavy")
                if not rows[bucket]["sparse"]
                and rows[bucket]["edge_per_initial_wager"] is not None
            ]
            neutral = rows["neutral"]["edge_per_initial_wager"]
            low_heavy_vs_neutral[str(box_count)] = (
                _mean(low_values) > neutral
                if low_values and neutral is not None
                else None
            )
        aggregate_marginals = [
            row
            for row in marginal_rows
            if row["source"] == source_name and row["seed_or_aggregate"] == "aggregate"
        ]
        non_sparse_marginal_exclusions = [
            {
                "state_bucket": row["state_bucket"],
                "added_box_number": row["added_box_number"],
                "marginal_return_per_added_box": row["marginal_return_per_added_box"],
                "seed_marginal_net_student_t_95_ci": row[
                    "seed_marginal_net_student_t_95_ci"
                ],
            }
            for row in aggregate_marginals
            if not row["sparse"]
            and _interval_excludes_zero(row.get("seed_marginal_net_student_t_95_ci"))
        ]
        checks[source_name] = {
            "edge_non_decreasing_from_high_heavy_to_low_heavy_by_box": ordered_by_box,
            "mean_low_heavy_edge_above_neutral_by_box": low_heavy_vs_neutral,
            "cells_with_seed_ci_excluding_zero": sum(
                _interval_excludes_zero(row.get("seed_edge_student_t_95_ci"))
                for row in source_rows
            ),
            "non_sparse_cells_with_seed_ci_excluding_zero": sum(
                not row["sparse"]
                and _interval_excludes_zero(row.get("seed_edge_student_t_95_ci"))
                for row in source_rows
            ),
            "non_sparse_marginal_cells_with_seed_ci_excluding_zero": (
                non_sparse_marginal_exclusions
            ),
            "sparse_aggregate_cells": sum(row["sparse"] for row in source_rows),
            "sparse_state_buckets": [
                row["state_bucket"]
                for row in frequency_rows
                if row["source"] == source_name
                and row["seed_or_aggregate"] == "aggregate"
                and row["sparse"]
            ],
            "maximum_dealer_upcard_rate_range_across_box_counts": (
                _maximum_dealer_rate_range(source_rows)
            ),
        }
    return checks


def _summary_markdown(
    config: MultiBoxCounterfactualConfig,
    frequency_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    marginal_rows: Sequence[Mapping[str, Any]],
    signal_checks: Mapping[str, Any],
    plot_paths: Mapping[str, str],
) -> str:
    lines = [
        "# Multi-Box Counterfactual Action Values",
        "",
        "This is a counterfactual next-round action-value experiment.",
        "",
        "It tests whether observable discard-rack composition predicts the next",
        "round and whether changing the number of active boxes changes the value",
        "captured from that state.",
        "",
        "It does not implement or validate an advantage strategy.",
        "",
        "## State Frequency",
        "",
        "| State bucket | Physical IID states | One2Six states |",
        "|---|---:|---:|",
    ]
    aggregate_frequency = {
        (row["source"], row["state_bucket"]): row
        for row in frequency_rows
        if row["seed_or_aggregate"] == "aggregate"
    }
    for bucket in STATE_BUCKETS:
        physical = aggregate_frequency[("physical_iid", bucket)]
        one2six = aggregate_frequency[("one2six", bucket)]
        lines.append(
            f"| {_bucket_label(bucket)} | {_count_with_sparse(physical)} | "
            f"{_count_with_sparse(one2six)} |"
        )

    for source_name in SOURCE_NAMES:
        lines.extend(
            [
                "",
                f"## {_source_label(source_name)} State-Conditioned Edge",
                "",
                _wide_table_header("State bucket", config.box_counts, "box"),
                _wide_table_separator(len(config.box_counts) + 1),
            ]
        )
        for bucket in STATE_BUCKETS:
            values = []
            for box_count in config.box_counts:
                row = _find_row(
                    action_rows,
                    source=source_name,
                    state_bucket=bucket,
                    box_count=box_count,
                )
                values.append(
                    _format_percent(row["edge_per_initial_wager"], row["sparse"])
                )
            lines.append(f"| {_bucket_label(bucket)} | " + " | ".join(values) + " |")

    aggregate_marginals = [
        row for row in marginal_rows if row["seed_or_aggregate"] == "aggregate"
    ]
    added_boxes = tuple(
        box_count
        for box_count in config.box_counts
        if box_count >= 2 and box_count - 1 in config.box_counts
    )
    for source_name in SOURCE_NAMES:
        lines.extend(
            [
                "",
                f"## {_source_label(source_name)} Paired Marginal Box Return",
                "",
                _wide_table_header("State bucket", added_boxes, "add box"),
                _wide_table_separator(len(added_boxes) + 1),
            ]
        )
        for bucket in STATE_BUCKETS:
            values = []
            for added_box in added_boxes:
                row = _find_row(
                    aggregate_marginals,
                    source=source_name,
                    state_bucket=bucket,
                    added_box_number=added_box,
                )
                values.append(
                    _format_percent(row["marginal_return_per_added_box"], row["sparse"])
                )
            lines.append(f"| {_bucket_label(bucket)} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Signal Checks",
            "",
            (
                "All checks are descriptive. A favourable cell is not a frozen policy "
                "and is not out-of-sample evidence."
            ),
            "",
        ]
    )
    for source_name in SOURCE_NAMES:
        source_checks = signal_checks[source_name]
        lines.extend(
            [
                f"- {_source_label(source_name)} monotonic by box: "
                f"{source_checks['edge_non_decreasing_from_high_heavy_to_low_heavy_by_box']}",
                f"- {_source_label(source_name)} low-heavy above neutral: "
                f"{source_checks['mean_low_heavy_edge_above_neutral_by_box']}",
                f"- {_source_label(source_name)} non-sparse edge cells with seed CI "
                "excluding zero: "
                f"{source_checks['non_sparse_cells_with_seed_ci_excluding_zero']}",
                f"- {_source_label(source_name)} non-sparse marginal cells with seed "
                "CI excluding zero: "
                f"{source_checks['non_sparse_marginal_cells_with_seed_ci_excluding_zero']}",
                f"- {_source_label(source_name)} sparse aggregate cells: "
                f"{source_checks['sparse_aggregate_cells']}",
                f"- {_source_label(source_name)} maximum dealer-upcard rate range "
                "across box counts: "
                f"{source_checks['maximum_dealer_upcard_rate_range_across_box_counts']:.4%}",
            ]
        )
    lines.extend(["", "## Plots", ""])
    for label, path in plot_paths.items():
        lines.append(f"- [{label}]({path})")
    lines.extend(
        [
            "",
            "`*` marks estimates with fewer than "
            f"{SPARSE_STATE_THRESHOLD} sampled states.",
            "",
        ]
    )
    return "\n".join(lines)


def _uncertainty_fields(summary: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_contributing_seeds": summary["independent_seed_runs"],
        f"{prefix}_mean": summary["mean"],
        f"{prefix}_sample_standard_deviation": summary["sample_standard_deviation"],
        f"{prefix}_standard_error": summary["standard_error"],
        f"{prefix}_student_t_95_ci": summary["student_t_95_ci"],
        f"{prefix}_minimum": summary["minimum"],
        f"{prefix}_maximum": summary["maximum"],
    }


def _category_rate(
    group: Sequence[BranchResult], category: DealerUpcardCategory
) -> float | None:
    return _rate_or_none(
        sum(branch.dealer_upcard_category == category for branch in group), len(group)
    )


def _position_category_rate(
    positions: Sequence[BoxPositionResult],
    field_name: str,
    category: CardCategory,
) -> float | None:
    return _rate_or_none(
        sum(getattr(position, field_name) == category for position in positions),
        len(positions),
    )


def _maximum_dealer_rate_range(rows: Sequence[Mapping[str, Any]]) -> float:
    maximum = 0.0
    for bucket in STATE_BUCKETS:
        bucket_rows = [
            row for row in rows if row["state_bucket"] == bucket and not row["sparse"]
        ]
        for field_name in (
            "dealer_upcard_ace_rate",
            "dealer_upcard_ten_value_rate",
            "dealer_upcard_low_rate",
            "dealer_upcard_neutral_rate",
        ):
            values = [
                row[field_name] for row in bucket_rows if row[field_name] is not None
            ]
            if values:
                maximum = max(maximum, max(values) - min(values))
    return maximum


def _interval_excludes_zero(interval: object) -> bool:
    if not isinstance(interval, list) or len(interval) != 2:
        return False
    return bool(interval[0] > 0 or interval[1] < 0)


def _find_row(
    rows: Sequence[Mapping[str, Any]], **matches: object
) -> Mapping[str, Any]:
    matching = [
        row
        for row in rows
        if all(row.get(field_name) == value for field_name, value in matches.items())
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one summary row for {matches!r}.")
    return matching[0]


def _wide_table_header(label: str, columns: Sequence[int], prefix: str) -> str:
    return f"| {label} | " + " | ".join(f"{prefix} {value}" for value in columns) + " |"


def _wide_table_separator(column_count: int) -> str:
    return "|---" + "|---:" * (column_count - 1) + "|"


def _count_with_sparse(row: Mapping[str, Any]) -> str:
    suffix = "*" if row["sparse"] else ""
    return f"{row['state_count']:,}{suffix}"


def _format_percent(value: float | None, sparse: bool) -> str:
    if value is None:
        return "n/a*" if sparse else "n/a"
    suffix = "*" if sparse else ""
    return f"{value:.3%}{suffix}"


def _bucket_label(bucket: str) -> str:
    return bucket.replace("_", " ").title()


def _source_label(source_name: str) -> str:
    return "Physical IID" if source_name == "physical_iid" else "One2Six"


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(numerator: float | int, denominator: float | int) -> float:
    return numerator / denominator if denominator else 0.0


def _rate_or_none(numerator: float | int, denominator: float | int) -> float | None:
    return numerator / denominator if denominator else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
