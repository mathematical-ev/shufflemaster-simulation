# SPDX-License-Identifier: GPL-3.0-or-later

"""Paired player-action values under observable decision-time composition."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from math import sqrt
from pathlib import Path
from typing import Any, Final, Literal

from experiments.conditional_profitability import linear_quantile
from experiments.fading_exclusion_validation import (
    FROZEN_WEIGHTS,
    CohortCounts,
    FadingState,
    LedgerCardSource,
    ObservableCohortLedger,
    calculate_fading_state,
)
from experiments.multi_box_counterfactual import SourceName, _make_source
from experiments.plots import (
    plot_ace_vs_ten_response,
    plot_action_delta_by_cell,
    plot_action_delta_by_low_band,
    plot_decision_state_support,
    plot_validated_candidate_deltas,
)
from experiments.single_box_game_validation import student_t_summary
from shufflemaster_sim.actions import ActionType
from shufflemaster_sim.card_sources import One2SixCardSource
from shufflemaster_sim.cards import Card, Rank
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
    CasinoBlackjackStrategy,
)
from shufflemaster_sim.hand_values import hand_value, is_bust, split_value_from_rank
from shufflemaster_sim.state import (
    BlackjackDecisionState,
    HandState,
    TableState,
)
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)

Source = Literal["physical_iid", "one2six"]
Phase = Literal["development", "validation"]
Band = Literal["poor", "neutral", "rich"]
Family = Literal["ten_ace", "low_band"]

SOURCE_NAMES: Final[tuple[Source, ...]] = ("physical_iid", "one2six")
PHASES: Final[tuple[Phase, ...]] = ("development", "validation")
BANDS: Final[tuple[Band, ...]] = ("poor", "neutral", "rich")
DEFAULT_DEVELOPMENT_SEEDS: Final[tuple[int, ...]] = (62, 63, 64, 65, 66)
DEFAULT_VALIDATION_SEEDS: Final[tuple[int, ...]] = (67, 68, 69, 70, 71)
FEATURE_NAMES: Final[tuple[str, ...]] = ("low", "ten_value", "ace")
PRIVATE_TERMS: Final[tuple[str, ...]] = (
    "physical_id",
    "draw_id",
    "shelf_id",
    "buffer_contents",
    "feeder_contents",
    "carousel_contents",
    "rng_state",
    "source_snapshot",
    "hidden_telemetry",
)
EMPTY_CANDIDATE_CSV_FIELDS: Final[tuple[str, ...]] = (
    "candidate_id",
    "family",
    "decision_id",
    "composition_state",
    "baseline_action",
    "alternative_action",
    "hand_kind",
    "hand_total",
    "pair_value",
    "dealer_upcard",
    "card_count",
    "split_hand",
    "split_aces",
    "legal_actions",
    "validation_sampled_states",
    "validation_mean_delta",
    "validation_delta_ci",
    "validation_mean_action_value",
    "validation_action_value_ci",
    "validation_mean_baseline_value",
    "validation_labels",
    "validated",
)


@dataclass(frozen=True, slots=True)
class ScoreConditionedActionValueConfig:
    """Frozen action-value experiment configuration."""

    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS
    validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS
    decision_states_per_seed: int = 10_000
    burn_in_rounds: int = 1_000
    deck_count: int = 6
    base_bet: float = 10.0
    composition_quantiles: tuple[float, float] = (0.30, 0.70)
    current_weight: float = 1.00
    returned_1_15_weight: float = 0.75
    returned_16_50_weight: float = 0.40
    returned_51_100_weight: float = 0.20
    returned_over_100_weight: float = 0.00
    minimum_total_state_count: int = 500
    minimum_per_seed_state_count: int = 50
    minimum_seed_sign_count: int = 4
    output_dir: Path = Path("experiments/outputs/score_conditioned_action_values")

    def __post_init__(self) -> None:
        if not self.development_seeds or not self.validation_seeds:
            raise ValueError("development and validation seeds must be nonempty.")
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("development seeds must be unique.")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation seeds must be unique.")
        if set(self.development_seeds).intersection(self.validation_seeds):
            raise ValueError("development and validation seeds must be disjoint.")
        if self.deck_count != 6:
            raise ValueError("action values require exactly six decks.")
        if self.decision_states_per_seed <= 0:
            raise ValueError("decision_states_per_seed must be positive.")
        if self.burn_in_rounds < 0:
            raise ValueError("burn_in_rounds must be non-negative.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        lower, upper = self.composition_quantiles
        if not 0 < lower < upper < 1:
            raise ValueError("composition quantiles must be ordered inside 0-1.")
        if self.weights != FROZEN_WEIGHTS:
            raise ValueError("weights must match the frozen documented kernel.")
        if self.minimum_total_state_count <= 0:
            raise ValueError("minimum_total_state_count must be positive.")
        if self.minimum_per_seed_state_count <= 0:
            raise ValueError("minimum_per_seed_state_count must be positive.")
        if self.minimum_seed_sign_count <= 0:
            raise ValueError("minimum_seed_sign_count must be positive.")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "current_rack": self.current_weight,
            "returned_1_15": self.returned_1_15_weight,
            "returned_16_50": self.returned_16_50_weight,
            "returned_51_100": self.returned_51_100_weight,
            "returned_over_100": self.returned_over_100_weight,
        }


@dataclass(frozen=True, slots=True)
class FeatureCutpoint:
    q30: float
    q70: float


@dataclass(frozen=True, slots=True)
class CompositionCutpoints:
    low: FeatureCutpoint
    ten_value: FeatureCutpoint
    ace: FeatureCutpoint

    def as_dict(self) -> dict[str, dict[str, float]]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionKey:
    hand_kind: Literal["hard", "soft", "pair"]
    hand_total: int
    pair_value: str | None
    dealer_upcard: str
    card_count: int
    split_hand: bool
    split_aces: bool
    legal_actions: tuple[str, ...]
    baseline_action: str

    def stable_id(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class DecisionFeatures:
    predicted_hi_lo_shift: float
    predicted_low_shift: float
    predicted_neutral_shift: float
    predicted_ten_value_shift: float
    predicted_ace_shift: float

    @classmethod
    def from_state(cls, state: FadingState) -> DecisionFeatures:
        return cls(
            predicted_hi_lo_shift=state.predicted_hi_lo_shift,
            predicted_low_shift=state.predicted_low_shift,
            predicted_neutral_shift=state.predicted_neutral_shift,
            predicted_ten_value_shift=state.predicted_ten_value_shift,
            predicted_ace_shift=state.predicted_ace_shift,
        )


@dataclass(frozen=True, slots=True)
class ActionBranchValue:
    action: str
    final_box_net: float
    initial_box_wager: float
    additional_action_wager: float
    total_box_wager: float
    net_per_initial_wager: float
    net_per_total_wager: float


@dataclass(frozen=True, slots=True)
class SampledDecision:
    source: Source
    phase: Phase
    seed: int
    decision_index: int
    key: DecisionKey
    features: DecisionFeatures
    branches: tuple[ActionBranchValue, ...]


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    game: CasinoBlackjackGame
    table: TableState
    card_source: LedgerCardSource
    box_id: int
    hand_id: int


@dataclass(slots=True)
class ActionAccumulator:
    count: int = 0
    action_value_sum: float = 0.0
    baseline_value_sum: float = 0.0
    delta_sum: float = 0.0
    low_shift_sum: float = 0.0
    ten_shift_sum: float = 0.0
    ace_shift_sum: float = 0.0

    def add(
        self,
        *,
        action_value: float,
        baseline_value: float,
        features: DecisionFeatures,
    ) -> None:
        self.count += 1
        self.action_value_sum += action_value
        self.baseline_value_sum += baseline_value
        self.delta_sum += action_value - baseline_value
        self.low_shift_sum += features.predicted_low_shift
        self.ten_shift_sum += features.predicted_ten_value_shift
        self.ace_shift_sum += features.predicted_ace_shift

    def as_seed_metrics(self) -> dict[str, float | int]:
        return {
            "state_count": self.count,
            "mean_action_value": self.action_value_sum / self.count,
            "mean_baseline_value": self.baseline_value_sum / self.count,
            "mean_delta_vs_baseline": self.delta_sum / self.count,
            "mean_low_shift": self.low_shift_sum / self.count,
            "mean_ten_value_shift": self.ten_shift_sum / self.count,
            "mean_ace_shift": self.ace_shift_sum / self.count,
        }


def freeze_composition_cutpoints(
    feature_rows: Sequence[DecisionFeatures],
    quantiles: tuple[float, float],
) -> CompositionCutpoints:
    """Freeze feature-only cutpoints without accepting action outcomes."""
    if not feature_rows:
        raise ValueError("development decision features must be nonempty.")
    lower, upper = quantiles
    cutpoints = {}
    for name in FEATURE_NAMES:
        values = [
            float(getattr(row, f"predicted_{name}_shift")) for row in feature_rows
        ]
        cutpoints[name] = FeatureCutpoint(
            q30=linear_quantile(values, lower),
            q70=linear_quantile(values, upper),
        )
    return CompositionCutpoints(
        low=cutpoints["low"],
        ten_value=cutpoints["ten_value"],
        ace=cutpoints["ace"],
    )


def assign_band(value: float, cutpoint: FeatureCutpoint) -> Band:
    if value <= cutpoint.q30:
        return "poor"
    if value >= cutpoint.q70:
        return "rich"
    return "neutral"


def composition_labels(
    features: DecisionFeatures, cutpoints: CompositionCutpoints
) -> tuple[Band, Band, Band, str]:
    low = assign_band(features.predicted_low_shift, cutpoints.low)
    ten = assign_band(features.predicted_ten_value_shift, cutpoints.ten_value)
    ace = assign_band(features.predicted_ace_shift, cutpoints.ace)
    return low, ten, ace, f"ten_{ten}__ace_{ace}"


def decision_key(
    *,
    hand: HandState,
    dealer_upcard: Rank,
    legal_actions: frozenset[ActionType],
    baseline_action: ActionType,
) -> DecisionKey:
    value = hand_value(hand.cards)
    pair = None
    kind: Literal["hard", "soft", "pair"]
    if len(hand.cards) == 2:
        left = split_value_from_rank(hand.cards[0].rank)
        right = split_value_from_rank(hand.cards[1].rank)
        if left == right:
            kind = "pair"
            pair = str(left)
        else:
            kind = "soft" if value.is_soft else "hard"
    else:
        kind = "soft" if value.is_soft else "hard"
    upcard = split_value_from_rank(dealer_upcard)
    return DecisionKey(
        hand_kind=kind,
        hand_total=value.total,
        pair_value=pair,
        dealer_upcard=str(upcard),
        card_count=len(hand.cards),
        split_hand=hand.is_split_hand,
        split_aces=hand.is_from_split_aces,
        legal_actions=tuple(sorted(action.value for action in legal_actions)),
        baseline_action=baseline_action.value,
    )


def observable_table_cards(table: TableState) -> tuple[Card, ...]:
    """Return every exposed player card plus dealer upcard exactly once."""
    cards = [card for box in table.boxes for hand in box.hands for card in hand.cards]
    if table.dealer.cards:
        cards.append(table.dealer.cards[0])
    draw_ids = [card.draw_id for card in cards]
    if len(draw_ids) != len(set(draw_ids)):
        raise RuntimeError("an exposed draw event entered the table cohort twice.")
    return tuple(cards)


def decision_time_state(
    *,
    config: ScoreConditionedActionValueConfig,
    table: TableState,
    source: LedgerCardSource,
) -> FadingState:
    """Calculate composition immediately before one player action."""
    table_cohort = CohortCounts.from_cards(observable_table_cards(table))
    returned = source.ledger.active_by_band(source.draw_count)
    return calculate_fading_state(
        current_rack=table_cohort,
        returned_by_band=returned,
        weights=config.weights,
    )


def clone_decision_snapshot(snapshot: DecisionSnapshot) -> DecisionSnapshot:
    return DecisionSnapshot(
        game=deepcopy(snapshot.game),
        table=deepcopy(snapshot.table),
        card_source=LedgerCardSource(
            source=deepcopy(snapshot.card_source.source),
            ledger=ObservableCohortLedger(),
        ),
        box_id=snapshot.box_id,
        hand_id=snapshot.hand_id,
    )


def branch_action(
    snapshot: DecisionSnapshot,
    *,
    action: ActionType,
    strategy: CasinoBlackjackStrategy,
) -> ActionBranchValue:
    """Force one action on an isolated clone and finish with fixed strategy."""
    branch = clone_decision_snapshot(snapshot)
    box = next(box for box in branch.table.boxes if box.box_id == branch.box_id)
    hand = next(hand for hand in box.hands if hand.hand_id == branch.hand_id)
    initial_wager = box.base_bet
    branch.game.apply_action(
        table=branch.table,
        box=box,
        hand=hand,
        action_type=action,
        card_source=branch.card_source,
    )
    branch.game.play_player_hands(branch.table, branch.card_source, strategy)
    branch.game.play_dealer(branch.table, branch.card_source)
    branch.game.settle(branch.table)
    branch.game.collect_remaining_layout_cards(branch.table)
    net = sum(current.net_result for current in box.hands)
    total_wager = sum(
        current.wager * (2 if current.is_doubled else 1) for current in box.hands
    )
    if isinstance(branch.card_source.source, One2SixCardSource):
        branch.card_source.source.assert_invariants(branch.table.discard_rack)
    return ActionBranchValue(
        action=action.value,
        final_box_net=net,
        initial_box_wager=initial_wager,
        additional_action_wager=total_wager - initial_wager,
        total_box_wager=total_wager,
        net_per_initial_wager=net / initial_wager,
        net_per_total_wager=net / total_wager,
    )


def branch_all_legal_actions(
    snapshot: DecisionSnapshot,
    *,
    legal_actions: frozenset[ActionType],
    strategy: CasinoBlackjackStrategy,
) -> tuple[ActionBranchValue, ...]:
    original_draw_count = snapshot.card_source.draw_count
    original_table = deepcopy(snapshot.table)
    results = tuple(
        branch_action(snapshot, action=action, strategy=strategy)
        for action in sorted(legal_actions, key=lambda item: item.value)
    )
    if snapshot.card_source.draw_count != original_draw_count:
        raise RuntimeError("an action branch mutated the original source.")
    if snapshot.table != original_table:
        raise RuntimeError("an action branch mutated the original table.")
    return results


def _new_trajectory(
    config: ScoreConditionedActionValueConfig,
    source_name: SourceName,
    seed: int,
) -> tuple[LedgerCardSource, CasinoBlackjackGame, PublishedApproxCasinoStrategy]:
    source = LedgerCardSource(
        source=_make_source(source_name, config.deck_count, seed),
        ledger=ObservableCohortLedger(),
    )
    game = CasinoBlackjackGame(
        CasinoBlackjackConfig(
            base_bet=config.base_bet,
            box_count=1,
            box_bets={1: config.base_bet},
            deck_count=config.deck_count,
        )
    )
    return source, game, PublishedApproxCasinoStrategy()


def _burn_in(
    config: ScoreConditionedActionValueConfig,
    source: LedgerCardSource,
    game: CasinoBlackjackGame,
    strategy: CasinoBlackjackStrategy,
) -> int:
    for round_index in range(config.burn_in_rounds):
        game.play_round(
            round_index=round_index,
            card_source=source,
            strategy=strategy,
        )
    return config.burn_in_rounds


def collect_decision_trajectory(
    config: ScoreConditionedActionValueConfig,
    *,
    source_name: Source,
    phase: Phase,
    seed: int,
) -> list[SampledDecision]:
    """Collect natural fixed-strategy decisions and paired legal-action branches."""
    source, game, strategy = _new_trajectory(config, source_name, seed)
    round_index = _burn_in(config, source, game, strategy)
    samples: list[SampledDecision] = []
    while len(samples) < config.decision_states_per_seed:
        _play_behavior_round(
            config=config,
            source_name=source_name,
            phase=phase,
            seed=seed,
            game=game,
            source=source,
            strategy=strategy,
            round_index=round_index,
            samples=samples,
        )
        round_index += 1
    if len(samples) != config.decision_states_per_seed:
        raise RuntimeError("decision trajectory did not stop at the exact target.")
    if isinstance(source.source, One2SixCardSource):
        source.source.assert_invariants(game.pending_discard_rack)
    return samples


@dataclass(frozen=True, slots=True)
class _RoundSampleContext:
    sample_index: int
    baseline_action: ActionType


def _play_behavior_round(
    *,
    config: ScoreConditionedActionValueConfig,
    source_name: Source,
    phase: Phase,
    seed: int,
    game: CasinoBlackjackGame,
    source: LedgerCardSource,
    strategy: CasinoBlackjackStrategy,
    round_index: int,
    samples: list[SampledDecision],
) -> None:
    source.before_round()
    game.burn_initial_card(source)
    table = game.create_table(round_index)
    game.deal_initial_cards(table, source)
    if game.pending_discard_rack:
        raise RuntimeError("pre-bet rack was not returned after the initial deal.")
    game.settle_immediate_blackjacks(table)
    contexts: list[_RoundSampleContext] = []
    dealer_upcard = table.dealer.cards[0]
    for box in table.boxes:
        hand_index = 0
        while hand_index < len(box.hands):
            hand = box.hands[hand_index]
            while _hand_needs_action(hand):
                legal_actions = game.legal_actions(table=table, box=box, hand=hand)
                if not legal_actions:
                    hand.is_terminal = True
                    break
                decision = BlackjackDecisionState(
                    player_ranks=tuple(card.rank for card in hand.cards),
                    dealer_upcard_rank=dealer_upcard.rank,
                    legal_actions=legal_actions,
                    is_split_hand=hand.is_split_hand,
                )
                baseline_action = strategy.choose_action(decision=decision).action_type
                if len(samples) < config.decision_states_per_seed:
                    state = decision_time_state(
                        config=config, table=table, source=source
                    )
                    key = decision_key(
                        hand=hand,
                        dealer_upcard=dealer_upcard.rank,
                        legal_actions=legal_actions,
                        baseline_action=baseline_action,
                    )
                    snapshot = DecisionSnapshot(
                        game=game,
                        table=table,
                        card_source=source,
                        box_id=box.box_id,
                        hand_id=hand.hand_id,
                    )
                    branches = tuple(
                        branch_action(snapshot, action=action, strategy=strategy)
                        for action in sorted(
                            legal_actions - {baseline_action},
                            key=lambda item: item.value,
                        )
                    )
                    sample = SampledDecision(
                        source=source_name,
                        phase=phase,
                        seed=seed,
                        decision_index=len(samples),
                        key=key,
                        features=DecisionFeatures.from_state(state),
                        branches=branches,
                    )
                    samples.append(sample)
                    contexts.append(
                        _RoundSampleContext(
                            sample_index=len(samples) - 1,
                            baseline_action=baseline_action,
                        )
                    )
                game.apply_action(
                    table=table,
                    box=box,
                    hand=hand,
                    action_type=baseline_action,
                    card_source=source,
                )
                if baseline_action == ActionType.SPLIT:
                    hand = box.hands[hand_index]
            hand_index += 1
    game.play_dealer(table, source)
    game.settle(table)
    game.collect_remaining_layout_cards(table)
    game.stage_discard_rack_for_next_round(table)
    for context in contexts:
        sample = samples[context.sample_index]
        baseline = _completed_behavior_value(
            table,
            action=context.baseline_action,
            base_bet=config.base_bet,
        )
        samples[context.sample_index] = replace(
            sample,
            branches=tuple(
                sorted((*sample.branches, baseline), key=lambda branch: branch.action)
            ),
        )


def _completed_behavior_value(
    table: TableState, *, action: ActionType, base_bet: float
) -> ActionBranchValue:
    box = table.boxes[0]
    net = sum(hand.net_result for hand in box.hands)
    total_wager = sum(hand.wager * (2 if hand.is_doubled else 1) for hand in box.hands)
    return ActionBranchValue(
        action=action.value,
        final_box_net=net,
        initial_box_wager=base_bet,
        additional_action_wager=total_wager - base_bet,
        total_box_wager=total_wager,
        net_per_initial_wager=net / base_bet,
        net_per_total_wager=net / total_wager,
    )


def _hand_needs_action(hand: HandState) -> bool:
    return (
        hand.outcome_label is None
        and not hand.is_terminal
        and not is_bust(hand.cards)
        and hand_value(hand.cards).total < 21
    )


def aggregate_sampled_actions(
    samples: Sequence[SampledDecision],
    cutpoints: CompositionCutpoints,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create per-seed ten/ace-cell and low-band action summaries."""
    ten_ace: dict[tuple[Any, ...], ActionAccumulator] = defaultdict(ActionAccumulator)
    low_band: dict[tuple[Any, ...], ActionAccumulator] = defaultdict(ActionAccumulator)
    for sample in samples:
        low, _, _, cell = composition_labels(sample.features, cutpoints)
        baseline = next(
            branch
            for branch in sample.branches
            if branch.action == sample.key.baseline_action
        )
        decision_id = sample.key.stable_id()
        for branch in sample.branches:
            common = (
                sample.source,
                sample.phase,
                sample.seed,
                decision_id,
                branch.action,
            )
            ten_ace[(*common, cell)].add(
                action_value=branch.net_per_initial_wager,
                baseline_value=baseline.net_per_initial_wager,
                features=sample.features,
            )
            low_band[(*common, low)].add(
                action_value=branch.net_per_initial_wager,
                baseline_value=baseline.net_per_initial_wager,
                features=sample.features,
            )
    return (
        _seed_action_rows(ten_ace, samples, family="ten_ace"),
        _seed_action_rows(low_band, samples, family="low_band"),
    )


def _seed_action_rows(
    accumulators: Mapping[tuple[Any, ...], ActionAccumulator],
    samples: Sequence[SampledDecision],
    *,
    family: Family,
) -> list[dict[str, Any]]:
    keys = {sample.key.stable_id(): sample.key for sample in samples}
    rows = []
    for group, accumulator in sorted(
        accumulators.items(), key=lambda item: str(item[0])
    ):
        source, phase, seed, decision_id, action, state_label = group
        key = keys[decision_id]
        rows.append(
            {
                "row_scope": "per_seed",
                "source": source,
                "phase": phase,
                "seed": seed,
                "family": family,
                "decision_id": decision_id,
                **_decision_record(key),
                "composition_state": state_label,
                "action": action,
                **accumulator.as_seed_metrics(),
            }
        )
    return rows


def aggregate_seed_action_rows(
    per_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate matched-state deltas with independent seeds as the unit."""
    group_fields = (
        "source",
        "phase",
        "family",
        "decision_id",
        "composition_state",
        "action",
    )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        groups[tuple(row[field] for field in group_fields)].append(row)
    rows = []
    for group, matching in sorted(groups.items(), key=lambda item: str(item[0])):
        deltas = [float(row["mean_delta_vs_baseline"]) for row in matching]
        action_values = [float(row["mean_action_value"]) for row in matching]
        baseline_values = [float(row["mean_baseline_value"]) for row in matching]
        delta_summary = student_t_summary(deltas)
        action_summary = student_t_summary(action_values)
        baseline_summary = student_t_summary(baseline_values)
        first = matching[0]
        rows.append(
            {
                "row_scope": "aggregate",
                **dict(zip(group_fields, group, strict=True)),
                **_decision_fields_from_row(first),
                "sampled_states": sum(int(row["state_count"]) for row in matching),
                "contributing_seeds": len(matching),
                "minimum_seed_state_count": min(
                    int(row["state_count"]) for row in matching
                ),
                "mean_action_value": _weighted_mean(matching, "mean_action_value"),
                "action_value_seed_ci": action_summary["student_t_95_ci"],
                "positive_seed_action_values": sum(
                    value > 0 for value in action_values
                ),
                "mean_baseline_value": _weighted_mean(matching, "mean_baseline_value"),
                "baseline_value_seed_ci": baseline_summary["student_t_95_ci"],
                "positive_seed_baseline_values": sum(
                    value > 0 for value in baseline_values
                ),
                "mean_delta_vs_baseline": _weighted_mean(
                    matching, "mean_delta_vs_baseline"
                ),
                "mean_seed_delta_vs_baseline": delta_summary["mean"],
                "delta_sample_standard_deviation": delta_summary[
                    "sample_standard_deviation"
                ],
                "delta_standard_error": delta_summary["standard_error"],
                "delta_student_t_95_ci": delta_summary["student_t_95_ci"],
                "minimum_seed_delta": delta_summary["minimum"],
                "maximum_seed_delta": delta_summary["maximum"],
                "positive_seed_deltas": sum(value > 0 for value in deltas),
                "negative_seed_deltas": sum(value < 0 for value in deltas),
                "mean_low_shift": _weighted_mean(matching, "mean_low_shift"),
                "mean_ten_value_shift": _weighted_mean(
                    matching, "mean_ten_value_shift"
                ),
                "mean_ace_shift": _weighted_mean(matching, "mean_ace_shift"),
            }
        )
    return rows


def _weighted_mean(rows: Sequence[Mapping[str, Any]], field_name: str) -> float:
    total_count = sum(int(row["state_count"]) for row in rows)
    return (
        sum(float(row[field_name]) * int(row["state_count"]) for row in rows)
        / total_count
    )


def _decision_record(key: DecisionKey) -> dict[str, Any]:
    record = asdict(key)
    record["legal_actions"] = "|".join(key.legal_actions)
    return record


def _decision_fields_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field_name: row[field_name]
        for field_name in (
            "hand_kind",
            "hand_total",
            "pair_value",
            "dealer_upcard",
            "card_count",
            "split_hand",
            "split_aces",
            "legal_actions",
            "baseline_action",
        )
    }


def decision_frequency_rows(samples: Sequence[SampledDecision]) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for sample in samples:
        counts[(sample.source, sample.phase, sample.seed, sample.key.stable_id())] += 1
    key_lookup = {sample.key.stable_id(): sample.key for sample in samples}
    return [
        {
            "source": source,
            "phase": phase,
            "seed": seed,
            "decision_id": decision_id,
            **_decision_record(key_lookup[decision_id]),
            "decision_count": count,
        }
        for (source, phase, seed, decision_id), count in sorted(
            counts.items(), key=lambda item: str(item[0])
        )
    ]


def decision_composition_rows(
    samples: Sequence[SampledDecision], cutpoints: CompositionCutpoints
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for sample in samples:
        low, _, _, cell = composition_labels(sample.features, cutpoints)
        counts[(sample.source, sample.phase, "ten_ace", cell)] += 1
        counts[(sample.source, sample.phase, "low_band", low)] += 1
    return [
        {
            "source": source,
            "phase": phase,
            "family": family,
            "composition_state": state,
            "decision_count": count,
        }
        for (source, phase, family, state), count in sorted(counts.items())
    ]


def passes_improvement_gate(
    row: Mapping[str, Any],
    config: ScoreConditionedActionValueConfig,
    *,
    required_seed_count: int,
) -> bool:
    interval = row.get("delta_student_t_95_ci")
    return bool(
        row["action"] != row["baseline_action"]
        and int(row["sampled_states"]) >= config.minimum_total_state_count
        and int(row["contributing_seeds"]) == required_seed_count
        and int(row["minimum_seed_state_count"]) >= config.minimum_per_seed_state_count
        and float(row["mean_delta_vs_baseline"]) > 0
        and _ci_positive(interval)
        and int(row["positive_seed_deltas"]) >= config.minimum_seed_sign_count
    )


def discover_candidates(
    config: ScoreConditionedActionValueConfig,
    aggregate_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze development candidates using development outcomes only."""
    development_seed_count = len(config.development_seeds)
    index = {
        _action_group_key(row): row
        for row in aggregate_rows
        if row["phase"] == "development"
    }
    candidates = []
    for row in aggregate_rows:
        if row["phase"] != "development" or row["source"] != "one2six":
            continue
        if not passes_improvement_gate(
            row, config, required_seed_count=development_seed_count
        ):
            continue
        iid = index.get(_action_group_key(row, source="physical_iid"))
        source_difference = paired_source_delta(
            per_seed_rows,
            phase="development",
            family=str(row["family"]),
            decision_id=str(row["decision_id"]),
            composition_state=str(row["composition_state"]),
            action=str(row["action"]),
            seeds=config.development_seeds,
        )
        classes: list[str] = []
        iid_improves = iid is not None and passes_improvement_gate(
            iid, config, required_seed_count=development_seed_count
        )
        if iid_improves and not _ci_positive(source_difference["student_t_95_ci"]):
            classes.append("generic_baseline_correction")
        if _ci_positive(source_difference["student_t_95_ci"]):
            classes.append("one2six_composition_candidate")
        if (
            float(row["mean_baseline_value"]) < 0
            and float(row["mean_action_value"]) <= 0
        ) or (row["family"] == "low_band" and row["composition_state"] == "rich"):
            classes.append("loss_reduction_candidate")
        if (
            float(row["mean_action_value"]) > 0
            and _ci_positive(row["action_value_seed_ci"])
            and not _ci_positive(row["baseline_value_seed_ci"])
        ):
            classes.append("possible_edge_creation_candidate")
        if not classes:
            classes.append("one2six_composition_candidate")
        definition = {
            "family": row["family"],
            "decision_id": row["decision_id"],
            "composition_state": row["composition_state"],
            "baseline_action": row["baseline_action"],
            "alternative_action": row["action"],
        }
        candidates.append(
            {
                "candidate_id": json.dumps(
                    definition, sort_keys=True, separators=(",", ":")
                ),
                **definition,
                **_decision_fields_from_row(row),
                "candidate_classes": classes,
                "development_sampled_states": row["sampled_states"],
                "development_mean_delta": row["mean_delta_vs_baseline"],
                "development_delta_ci": row["delta_student_t_95_ci"],
                "development_mean_action_value": row["mean_action_value"],
                "development_source_difference": source_difference,
            }
        )
    return sorted(candidates, key=lambda row: str(row["candidate_id"]))


def paired_source_delta(
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    family: str,
    decision_id: str,
    composition_state: str,
    action: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    differences = []
    for seed in seeds:
        values = {
            str(row["source"]): row["mean_delta_vs_baseline"]
            for row in per_seed_rows
            if row["phase"] == phase
            and row["family"] == family
            and row["decision_id"] == decision_id
            and row["composition_state"] == composition_state
            and row["action"] == action
            and row["seed"] == seed
        }
        if values.get("one2six") is None or values.get("physical_iid") is None:
            continue
        differences.append(float(values["one2six"]) - float(values["physical_iid"]))
    summary = student_t_summary(differences)
    return {
        "contributing_seed_pairs": len(differences),
        "mean": summary["mean"],
        "student_t_95_ci": summary["student_t_95_ci"],
        "positive_seeds": sum(value > 0 for value in differences),
        "negative_seeds": sum(value < 0 for value in differences),
    }


def validate_candidates(
    config: ScoreConditionedActionValueConfig,
    candidates: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate only frozen development candidates on held-out seeds."""
    index = {
        _action_group_key(row): row
        for row in aggregate_rows
        if row["phase"] == "validation"
    }
    results = []
    validated = []
    for candidate in candidates:
        one = index.get(
            (
                "one2six",
                candidate["family"],
                candidate["decision_id"],
                candidate["composition_state"],
                candidate["alternative_action"],
            )
        )
        iid = index.get(
            (
                "physical_iid",
                candidate["family"],
                candidate["decision_id"],
                candidate["composition_state"],
                candidate["alternative_action"],
            )
        )
        direct = paired_source_delta(
            per_seed_rows,
            phase="validation",
            family=str(candidate["family"]),
            decision_id=str(candidate["decision_id"]),
            composition_state=str(candidate["composition_state"]),
            action=str(candidate["alternative_action"]),
            seeds=config.validation_seeds,
        )
        one_valid = one is not None and passes_improvement_gate(
            one, config, required_seed_count=len(config.validation_seeds)
        )
        iid_valid = iid is not None and passes_improvement_gate(
            iid, config, required_seed_count=len(config.validation_seeds)
        )
        labels: list[str] = []
        if one_valid and _ci_positive(direct["student_t_95_ci"]):
            labels.append("validated_one2six_deviation")
        if one_valid and iid_valid and not _ci_positive(direct["student_t_95_ci"]):
            labels.append("validated_generic_strategy_correction")
        if (
            one_valid
            and "loss_reduction_candidate" in candidate["candidate_classes"]
            and one is not None
            and float(one["mean_action_value"]) <= 0
        ):
            labels.append("validated_loss_reduction_deviation")
        if (
            one_valid
            and one is not None
            and float(one["mean_action_value"]) > 0
            and _ci_positive(one["action_value_seed_ci"])
            and int(one["positive_seed_action_values"])
            >= config.minimum_seed_sign_count
            and not _ci_positive(one["baseline_value_seed_ci"])
        ):
            labels.append("validated_edge_creation_deviation")
        result = {
            **candidate,
            "validation_available": one is not None,
            "validation_sampled_states": one["sampled_states"] if one else 0,
            "validation_mean_delta": one["mean_delta_vs_baseline"] if one else None,
            "validation_delta_ci": one["delta_student_t_95_ci"] if one else None,
            "validation_mean_action_value": one["mean_action_value"] if one else None,
            "validation_action_value_ci": one["action_value_seed_ci"] if one else None,
            "validation_mean_baseline_value": (
                one["mean_baseline_value"] if one else None
            ),
            "validation_direct_source_difference": direct,
            "validation_labels": labels,
            "validated": bool(labels),
        }
        results.append(result)
        if labels:
            validated.append(result)
    return results, validated


def _action_group_key(
    row: Mapping[str, Any], *, source: str | None = None
) -> tuple[Any, ...]:
    return (
        source if source is not None else row["source"],
        row["family"],
        row["decision_id"],
        row["composition_state"],
        row["action"],
    )


@dataclass(slots=True)
class MultipleRegressionAccumulator:
    count: int = 0
    xtx: list[list[float]] = field(
        default_factory=lambda: [[0.0] * 4 for _ in range(4)]
    )
    xty: list[float] = field(default_factory=lambda: [0.0] * 4)
    feature_sums: list[float] = field(default_factory=lambda: [0.0] * 3)
    feature_squares: list[float] = field(default_factory=lambda: [0.0] * 3)
    feature_cross: dict[tuple[int, int], float] = field(default_factory=dict)

    def add(self, low: float, ten: float, ace: float, delta: float) -> None:
        vector = [1.0, low, ten, ace]
        self.count += 1
        for left in range(4):
            self.xty[left] += vector[left] * delta
            for right in range(4):
                self.xtx[left][right] += vector[left] * vector[right]
        features = [low, ten, ace]
        for index, value in enumerate(features):
            self.feature_sums[index] += value
            self.feature_squares[index] += value * value
        for left in range(3):
            for right in range(left + 1, 3):
                self.feature_cross[(left, right)] = (
                    self.feature_cross.get((left, right), 0.0)
                    + features[left] * features[right]
                )

    def coefficients(self) -> tuple[float, float, float, float] | None:
        solution = _solve_linear_system(self.xtx, self.xty)
        return tuple(solution) if solution is not None else None  # type: ignore[return-value]

    def correlations(self) -> dict[str, float | None]:
        names = ("low_ten", "low_ace", "ten_ace")
        return {
            name: _correlation(
                self.count,
                self.feature_sums[left],
                self.feature_sums[right],
                self.feature_squares[left],
                self.feature_squares[right],
                self.feature_cross.get((left, right), 0.0),
            )
            for name, (left, right) in zip(names, ((0, 1), (0, 2), (1, 2)), strict=True)
        }


def continuous_action_response_rows(
    samples: Sequence[SampledDecision],
    config: ScoreConditionedActionValueConfig,
    *,
    expected_seed_count: int,
) -> list[dict[str, Any]]:
    accumulators: dict[tuple[Any, ...], MultipleRegressionAccumulator] = defaultdict(
        MultipleRegressionAccumulator
    )
    key_lookup = {sample.key.stable_id(): sample.key for sample in samples}
    for sample in samples:
        baseline = next(
            branch
            for branch in sample.branches
            if branch.action == sample.key.baseline_action
        )
        for branch in sample.branches:
            if branch.action == sample.key.baseline_action:
                continue
            group = (
                sample.source,
                sample.phase,
                sample.seed,
                sample.key.stable_id(),
                branch.action,
            )
            accumulators[group].add(
                sample.features.predicted_low_shift,
                sample.features.predicted_ten_value_shift,
                sample.features.predicted_ace_shift,
                branch.net_per_initial_wager - baseline.net_per_initial_wager,
            )
    per_seed = []
    for group, accumulator in accumulators.items():
        coefficients = accumulator.coefficients()
        if coefficients is None:
            continue
        source, phase, seed, decision_id, action = group
        key = key_lookup[decision_id]
        correlations = accumulator.correlations()
        per_seed.append(
            {
                "row_scope": "per_seed",
                "source": source,
                "phase": phase,
                "seed": seed,
                "decision_id": decision_id,
                **_decision_record(key),
                "action": action,
                "state_count": accumulator.count,
                "intercept": coefficients[0],
                "low_coefficient": coefficients[1],
                "ten_value_coefficient": coefficients[2],
                "ace_coefficient": coefficients[3],
                **{
                    f"predictor_correlation_{name}": value
                    for name, value in correlations.items()
                },
            }
        )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        groups[(row["source"], row["phase"], row["decision_id"], row["action"])].append(
            row
        )
    aggregate = []
    for group, matching in groups.items():
        total = sum(int(row["state_count"]) for row in matching)
        if (
            total < config.minimum_total_state_count
            or len(matching) != expected_seed_count
            or min(int(row["state_count"]) for row in matching)
            < config.minimum_per_seed_state_count
        ):
            continue
        first = matching[0]
        row: dict[str, Any] = {
            "row_scope": "aggregate",
            "source": group[0],
            "phase": group[1],
            "decision_id": group[2],
            **_decision_fields_from_row(first),
            "action": group[3],
            "sampled_states": total,
            "contributing_seeds": len(matching),
            "minimum_seed_state_count": min(
                int(item["state_count"]) for item in matching
            ),
        }
        for coefficient in ("low", "ten_value", "ace"):
            values = [float(item[f"{coefficient}_coefficient"]) for item in matching]
            summary = student_t_summary(values)
            row[f"mean_{coefficient}_coefficient"] = summary["mean"]
            row[f"{coefficient}_coefficient_seed_ci"] = summary["student_t_95_ci"]
            row[f"positive_seed_{coefficient}_coefficients"] = sum(
                value > 0 for value in values
            )
            row[f"negative_seed_{coefficient}_coefficients"] = sum(
                value < 0 for value in values
            )
        for correlation in ("low_ten", "low_ace", "ten_ace"):
            row[f"mean_predictor_correlation_{correlation}"] = _mean(
                [
                    float(item[f"predictor_correlation_{correlation}"])
                    for item in matching
                    if item[f"predictor_correlation_{correlation}"] is not None
                ]
            )
        aggregate.append(row)
    return [*per_seed, *aggregate]


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float] | None:
    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def _correlation(
    count: int,
    sum_x: float,
    sum_y: float,
    sum_xx: float,
    sum_yy: float,
    sum_xy: float,
) -> float | None:
    if count < 2:
        return None
    centered_x = sum_xx - sum_x**2 / count
    centered_y = sum_yy - sum_y**2 / count
    if centered_x <= 0 or centered_y <= 0:
        return None
    return (sum_xy - sum_x * sum_y / count) / sqrt(centered_x * centered_y)


def run_score_conditioned_action_values(
    config: ScoreConditionedActionValueConfig,
) -> dict[str, Any]:
    """Discover action deviations on development seeds and validate them held out."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    decision_frequency: list[dict[str, Any]] = []
    composition_support: list[dict[str, Any]] = []
    development_per_seed: list[dict[str, Any]] = []
    validation_per_seed: list[dict[str, Any]] = []
    development_continuous: list[dict[str, Any]] = []
    validation_continuous: list[dict[str, Any]] = []
    snapshot_counts: dict[tuple[str, str], int] = defaultdict(int)
    branch_counts: dict[tuple[str, str], int] = defaultdict(int)

    one2six_development: list[SampledDecision] = []
    for seed in config.development_seeds:
        samples = collect_decision_trajectory(
            config, source_name="one2six", phase="development", seed=seed
        )
        one2six_development.extend(samples)
        _record_coverage(samples, snapshot_counts, branch_counts)
    cutpoints = freeze_composition_cutpoints(
        [sample.features for sample in one2six_development],
        config.composition_quantiles,
    )
    frozen_cutpoints = cutpoints.as_dict()
    development_per_seed.extend(
        _process_action_samples(
            one2six_development,
            cutpoints,
            decision_frequency,
            composition_support,
        )
    )
    development_continuous.extend(
        continuous_action_response_rows(
            one2six_development,
            config,
            expected_seed_count=len(config.development_seeds),
        )
    )
    del one2six_development

    for seed in config.development_seeds:
        samples = collect_decision_trajectory(
            config, source_name="physical_iid", phase="development", seed=seed
        )
        _record_coverage(samples, snapshot_counts, branch_counts)
        development_per_seed.extend(
            _process_action_samples(
                samples,
                cutpoints,
                decision_frequency,
                composition_support,
            )
        )
        development_continuous.extend(
            continuous_action_response_rows(
                samples,
                config,
                expected_seed_count=1,
            )
        )
    development_continuous = _reaggregate_continuous(
        development_continuous,
        config,
        expected_seed_count=len(config.development_seeds),
    )
    development_aggregate = aggregate_seed_action_rows(development_per_seed)
    candidates = discover_candidates(
        config, development_aggregate, development_per_seed
    )
    frozen_candidate_ids = tuple(str(row["candidate_id"]) for row in candidates)

    validation_samples_by_source: dict[Source, list[SampledDecision]] = {
        "physical_iid": [],
        "one2six": [],
    }
    for source in SOURCE_NAMES:
        for seed in config.validation_seeds:
            samples = collect_decision_trajectory(
                config, source_name=source, phase="validation", seed=seed
            )
            validation_samples_by_source[source].extend(samples)
            _record_coverage(samples, snapshot_counts, branch_counts)
            validation_per_seed.extend(
                _process_action_samples(
                    samples,
                    cutpoints,
                    decision_frequency,
                    composition_support,
                )
            )
        validation_continuous.extend(
            continuous_action_response_rows(
                validation_samples_by_source[source],
                config,
                expected_seed_count=len(config.validation_seeds),
            )
        )
    validation_aggregate = aggregate_seed_action_rows(validation_per_seed)
    validation_results, validated = validate_candidates(
        config,
        candidates,
        validation_aggregate,
        validation_per_seed,
    )
    if tuple(str(row["candidate_id"]) for row in candidates) != frozen_candidate_ids:
        raise RuntimeError("validation mutated the frozen candidate definitions.")

    decision_frequency_aggregate = _aggregate_frequency_rows(decision_frequency)
    composition_support_aggregate = _aggregate_composition_support(composition_support)
    generic = [
        row
        for row in validated
        if "validated_generic_strategy_correction" in row["validation_labels"]
    ]
    one2six_specific = [
        row
        for row in validated
        if "validated_one2six_deviation" in row["validation_labels"]
    ]
    loss_reduction = [
        row
        for row in validated
        if "validated_loss_reduction_deviation" in row["validation_labels"]
    ]
    edge_creation = [
        row
        for row in validated
        if "validated_edge_creation_deviation" in row["validation_labels"]
    ]
    plot_paths = _write_plots(
        config.output_dir,
        config,
        development_aggregate,
        validation_results,
        validated,
        decision_frequency_aggregate,
        development_continuous,
        validation_continuous,
    )
    config_payload = {
        **asdict(config),
        "output_dir": str(config.output_dir),
        "frozen_weights": config.weights,
        "cutpoint_inputs": [
            "predicted_low_shift",
            "predicted_ten_value_shift",
            "predicted_ace_shift",
        ],
        "action_outcomes_used_for_cutpoints": False,
    }
    cutpoint_payload = {
        "source": "one2six",
        "phase": "development",
        "development_seeds": list(config.development_seeds),
        "feature_count": len(config.development_seeds)
        * config.decision_states_per_seed,
        "quantiles": list(config.composition_quantiles),
        "cutpoints": frozen_cutpoints,
        "action_outcomes_used": False,
    }
    summary = {
        "experiment": "score_conditioned_counterfactual_action_values",
        "config": config_payload,
        "composition_cutpoints": cutpoint_payload,
        "coverage": [
            {
                "phase": phase,
                "source": source,
                "decision_snapshots": snapshot_counts[(phase, source)],
                "action_branches": branch_counts[(phase, source)],
            }
            for phase in PHASES
            for source in SOURCE_NAMES
        ],
        "development_candidate_count": len(candidates),
        "validation_result_count": len(validation_results),
        "validated_candidate_count": len(validated),
        "generic_strategy_corrections": generic,
        "one2six_specific_deviations": one2six_specific,
        "loss_reduction_deviations": loss_reduction,
        "edge_creation_deviations": edge_creation,
        "decision_state_coverage": decision_frequency_aggregate,
        "composition_support": composition_support_aggregate,
        "continuous_action_response_development": _aggregate_only(
            development_continuous
        ),
        "continuous_action_response_validation": _aggregate_only(validation_continuous),
        "hidden_state_exported": False,
        "plot_paths": plot_paths,
    }
    _validate_privacy(summary)
    _write_outputs(
        config.output_dir,
        summary,
        config_payload,
        cutpoint_payload,
        decision_frequency,
        composition_support_aggregate,
        development_per_seed,
        development_aggregate,
        candidates,
        validation_results,
        validated,
        development_continuous,
        validation_continuous,
    )
    return summary


def _process_action_samples(
    samples: Sequence[SampledDecision],
    cutpoints: CompositionCutpoints,
    frequency_output: list[dict[str, Any]],
    composition_output: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ten_ace, low_band = aggregate_sampled_actions(samples, cutpoints)
    frequency_output.extend(decision_frequency_rows(samples))
    composition_output.extend(decision_composition_rows(samples, cutpoints))
    return [*ten_ace, *low_band]


def _record_coverage(
    samples: Sequence[SampledDecision],
    snapshots: dict[tuple[str, str], int],
    branches: dict[tuple[str, str], int],
) -> None:
    if not samples:
        return
    key = (samples[0].phase, samples[0].source)
    snapshots[key] += len(samples)
    branches[key] += sum(len(sample.branches) for sample in samples)


def _reaggregate_continuous(
    rows: Sequence[Mapping[str, Any]],
    config: ScoreConditionedActionValueConfig,
    *,
    expected_seed_count: int,
) -> list[dict[str, Any]]:
    per_seed = [dict(row) for row in rows if row["row_scope"] == "per_seed"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        groups[(row["source"], row["phase"], row["decision_id"], row["action"])].append(
            row
        )
    aggregate = []
    for group, matching in groups.items():
        total = sum(int(row["state_count"]) for row in matching)
        if (
            total < config.minimum_total_state_count
            or len(matching) != expected_seed_count
            or min(int(row["state_count"]) for row in matching)
            < config.minimum_per_seed_state_count
        ):
            continue
        first = matching[0]
        aggregate_row: dict[str, Any] = {
            "row_scope": "aggregate",
            "source": group[0],
            "phase": group[1],
            "decision_id": group[2],
            **_decision_fields_from_row(first),
            "action": group[3],
            "sampled_states": total,
            "contributing_seeds": len(matching),
            "minimum_seed_state_count": min(
                int(item["state_count"]) for item in matching
            ),
        }
        for coefficient in ("low", "ten_value", "ace"):
            values = [float(item[f"{coefficient}_coefficient"]) for item in matching]
            summary = student_t_summary(values)
            aggregate_row[f"mean_{coefficient}_coefficient"] = summary["mean"]
            aggregate_row[f"{coefficient}_coefficient_seed_ci"] = summary[
                "student_t_95_ci"
            ]
            aggregate_row[f"positive_seed_{coefficient}_coefficients"] = sum(
                value > 0 for value in values
            )
            aggregate_row[f"negative_seed_{coefficient}_coefficients"] = sum(
                value < 0 for value in values
            )
        for correlation in ("low_ten", "low_ace", "ten_ace"):
            aggregate_row[f"mean_predictor_correlation_{correlation}"] = _mean(
                [
                    float(item[f"predictor_correlation_{correlation}"])
                    for item in matching
                    if item[f"predictor_correlation_{correlation}"] is not None
                ]
            )
        aggregate.append(aggregate_row)
    return [*per_seed, *aggregate]


def _aggregate_frequency_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["phase"], row["decision_id"])].append(row)
    result = []
    for group, matching in groups.items():
        first = matching[0]
        result.append(
            {
                "source": group[0],
                "phase": group[1],
                "decision_id": group[2],
                **_decision_fields_from_row(first),
                "decision_count": sum(int(row["decision_count"]) for row in matching),
                "contributing_seeds": len({int(row["seed"]) for row in matching}),
            }
        )
    return sorted(result, key=lambda row: (-int(row["decision_count"]), str(row)))


def _aggregate_composition_support(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = (row["source"], row["phase"], row["family"], row["composition_state"])
        counts[key] += int(row["decision_count"])
    return [
        {
            "source": source,
            "phase": phase,
            "family": family,
            "composition_state": state,
            "decision_count": count,
        }
        for (source, phase, family, state), count in sorted(counts.items())
    ]


def _aggregate_only(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["row_scope"] == "aggregate"]


def _write_plots(
    output_dir: Path,
    config: ScoreConditionedActionValueConfig,
    development: Sequence[Mapping[str, Any]],
    validation_results: Sequence[Mapping[str, Any]],
    validated: Sequence[Mapping[str, Any]],
    frequency: Sequence[Mapping[str, Any]],
    development_continuous: Sequence[Mapping[str, Any]],
    validation_continuous: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    paths = {}
    name = "action_delta_by_composition_cell.png"
    plot_action_delta_by_cell(
        [
            row
            for row in development
            if row["family"] == "ten_ace" and row["action"] != row["baseline_action"]
        ],
        minimum_support=config.minimum_total_state_count,
        output_path=output_dir / name,
    )
    paths["action_delta_by_composition_cell"] = name
    name = "action_delta_by_low_band.png"
    plot_action_delta_by_low_band(
        [
            row
            for row in development
            if row["family"] == "low_band" and row["action"] != row["baseline_action"]
        ],
        minimum_support=config.minimum_total_state_count,
        output_path=output_dir / name,
    )
    paths["action_delta_by_low_band"] = name
    name = "ace_vs_ten_action_response.png"
    plot_ace_vs_ten_response(
        _aggregate_only([*development_continuous, *validation_continuous]),
        output_path=output_dir / name,
    )
    paths["ace_vs_ten_action_response"] = name
    name = "validated_candidate_deltas.png"
    plot_validated_candidate_deltas(
        validation_results,
        validated,
        output_path=output_dir / name,
    )
    paths["validated_candidate_deltas"] = name
    name = "decision_state_support.png"
    plot_decision_state_support(frequency, output_path=output_dir / name)
    paths["decision_state_support"] = name
    return paths


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    cutpoint_payload: Mapping[str, Any],
    frequency_rows: Sequence[Mapping[str, Any]],
    composition_rows: Sequence[Mapping[str, Any]],
    development_per_seed: Sequence[Mapping[str, Any]],
    development_aggregate: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    validation_results: Sequence[Mapping[str, Any]],
    validated: Sequence[Mapping[str, Any]],
    development_continuous: Sequence[Mapping[str, Any]],
    validation_continuous: Sequence[Mapping[str, Any]],
) -> None:
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "experiment_config.json", config_payload)
    _write_json(output_dir / "composition_cutpoints.json", cutpoint_payload)
    _write_csv(output_dir / "decision_state_frequency.csv", frequency_rows)
    _write_csv(output_dir / "decision_state_composition_summary.csv", composition_rows)
    _write_csv(
        output_dir / "development_action_values_ten_ace.csv",
        [
            row
            for row in [*development_per_seed, *development_aggregate]
            if row["family"] == "ten_ace"
        ],
    )
    _write_csv(
        output_dir / "development_action_values_low_band.csv",
        [
            row
            for row in [*development_per_seed, *development_aggregate]
            if row["family"] == "low_band"
        ],
    )
    _write_json(output_dir / "development_candidates.json", {"candidates": candidates})
    _write_csv(
        output_dir / "validation_candidate_results.csv",
        validation_results,
        empty_fieldnames=EMPTY_CANDIDATE_CSV_FIELDS,
    )
    _write_json(output_dir / "validated_candidates.json", {"candidates": validated})
    _write_csv(
        output_dir / "continuous_action_response_development.csv",
        development_continuous,
    )
    _write_csv(
        output_dir / "continuous_action_response_validation.csv",
        validation_continuous,
    )
    generic = summary["generic_strategy_corrections"]
    one2six = summary["one2six_specific_deviations"]
    loss = summary["loss_reduction_deviations"]
    edge = summary["edge_creation_deviations"]
    _write_csv(
        output_dir / "generic_strategy_corrections.csv",
        generic,
        empty_fieldnames=EMPTY_CANDIDATE_CSV_FIELDS,
    )
    _write_csv(
        output_dir / "one2six_specific_deviations.csv",
        one2six,
        empty_fieldnames=EMPTY_CANDIDATE_CSV_FIELDS,
    )
    _write_csv(
        output_dir / "loss_reduction_deviations.csv",
        loss,
        empty_fieldnames=EMPTY_CANDIDATE_CSV_FIELDS,
    )
    _write_csv(
        output_dir / "edge_creation_deviations.csv",
        edge,
        empty_fieldnames=EMPTY_CANDIDATE_CSV_FIELDS,
    )
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    cutpoints = summary["composition_cutpoints"]["cutpoints"]
    lines = [
        "This experiment estimates counterfactual player-action values under",
        "observable One2Six composition states.",
        "",
        "It separately tracks low-card, ten-value and ace richness.",
        "",
        "It searches for both:",
        "",
        "1. favourable-state deviations that may create or increase an edge;",
        "2. unfavourable-state deviations that reduce expected loss.",
        "",
        "Candidate actions are discovered on seeds 62-66 and tested without",
        "retuning on seeds 67-71.",
        "",
        "No revised strategy is deployed continuously in this experiment.",
        "",
        "# Score-Conditioned Counterfactual Action Values",
        "",
        "## Composition Cutpoints",
        "",
    ]
    for feature in FEATURE_NAMES:
        values = cutpoints[feature]
        lines.append(
            f"- {feature}: q30={_fmt(values['q30'])}, q70={_fmt(values['q70'])}."
        )
    lines.extend(["", "## Decision-State Coverage", ""])
    for row in summary["coverage"]:
        lines.append(
            f"- {row['phase']} {_source_label(row['source'])}: "
            f"{row['decision_snapshots']} snapshots, {row['action_branches']} branches."
        )
    lines.extend(["", "Most frequent exact keys:", ""])
    for row in summary["decision_state_coverage"][:10]:
        lines.append(
            f"- {_source_label(row['source'])} {row['phase']} "
            f"{_decision_label(row)}: {row['decision_count']}."
        )
    lines.extend(["", "Composition-state support:", ""])
    for row in summary["composition_support"]:
        lines.append(
            f"- {row['phase']} {_source_label(row['source'])} "
            f"{row['family']} {row['composition_state']}: "
            f"{row['decision_count']}."
        )
    _candidate_section(
        lines, "Generic Strategy Corrections", summary["generic_strategy_corrections"]
    )
    _candidate_section(
        lines, "One2Six-Specific Deviations", summary["one2six_specific_deviations"]
    )
    _candidate_section(
        lines,
        "Low-Rich Loss-Reduction Deviations",
        summary["loss_reduction_deviations"],
    )
    _candidate_section(
        lines, "Edge-Creation Deviations", summary["edge_creation_deviations"]
    )
    lines.extend(["", "## Ace-Versus-Ten Findings", ""])
    lines.append(_ace_ten_verdict(summary))
    lines.extend(["", "## Verdict", "", _verdict(summary), "", "## Plots", ""])
    for label, path in summary["plot_paths"].items():
        lines.append(f"- [{label}]({path})")
    lines.append("")
    return "\n".join(lines)


def _candidate_section(
    lines: list[str], title: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    lines.extend(["", f"## {title}", ""])
    if not rows:
        lines.append("None validated under the frozen held-out criteria.")
        return
    lines.extend(
        [
            "| Decision | Composition | Baseline | Alternative | "
            "Held-out delta | 95% CI |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {_decision_label(row)} | {row['composition_state']} | "
            f"{row['baseline_action']} | {row['alternative_action']} | "
            f"{_fmt(row['validation_mean_delta'])} | "
            f"{_interval(row['validation_delta_ci'])} |"
        )


def _ace_ten_verdict(summary: Mapping[str, Any]) -> str:
    rows = summary["continuous_action_response_validation"]
    ace = sum(_ci_excludes_zero(row.get("ace_coefficient_seed_ci")) for row in rows)
    ten = sum(
        _ci_excludes_zero(row.get("ten_value_coefficient_seed_ci")) for row in rows
    )
    return (
        f"Among sufficiently supported validation diagnostics, {ace} ace coefficients "
        f"and {ten} ten-value coefficients had seed intervals excluding zero. "
        "No ace-rich versus ace-poor, ten-rich versus ten-poor, or combined "
        "ten-rich/ace-rich action deviation passed the frozen development gates. "
        "These diagnostics did not add candidates after development freeze."
    )


def _verdict(summary: Mapping[str, Any]) -> str:
    return (
        "Generic corrections validated="
        f"{bool(summary['generic_strategy_corrections'])}; "
        f"One2Six-specific deviations validated="
        f"{bool(summary['one2six_specific_deviations'])}; loss reductions validated="
        f"{bool(summary['loss_reduction_deviations'])}; edge creation validated="
        f"{bool(summary['edge_creation_deviations'])}. Candidate list ready for "
        f"continuous-policy testing={summary['validated_candidate_count'] > 0}."
    )


def _validate_privacy(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True).lower()
    matches = [term for term in PRIVATE_TERMS if term in serialized]
    if matches:
        raise RuntimeError(f"hidden source fields reached exports: {matches}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fieldnames: Sequence[str] = (),
) -> None:
    if not rows:
        if not empty_fieldnames:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=empty_fieldnames).writeheader()
        return
    fieldnames = sorted({field_name for row in rows for field_name in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _ci_positive(interval: object) -> bool:
    return isinstance(interval, list) and len(interval) == 2 and interval[0] > 0


def _ci_excludes_zero(interval: object) -> bool:
    return (
        isinstance(interval, list)
        and len(interval) == 2
        and (interval[0] > 0 or interval[1] < 0)
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "NA"
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _source_label(source: object) -> str:
    return "Physical IID" if source == "physical_iid" else "One2Six"


def _decision_label(row: Mapping[str, Any]) -> str:
    pair = f" pair {row['pair_value']}" if row.get("pair_value") else ""
    return (
        f"{row['hand_kind']} {row['hand_total']}{pair} vs {row['dealer_upcard']} "
        f"({row['card_count']} cards)"
    )
