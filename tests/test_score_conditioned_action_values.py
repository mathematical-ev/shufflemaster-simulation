# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy
from pathlib import Path

import pytest
from experiments.fading_exclusion_validation import (
    CohortCounts,
    LedgerCardSource,
    ObservableCohortLedger,
    ReturnedCohort,
)
from experiments.multi_box_counterfactual import _make_source
from experiments.score_conditioned_action_values import (
    DEFAULT_DEVELOPMENT_SEEDS,
    DEFAULT_VALIDATION_SEEDS,
    PRIVATE_TERMS,
    ActionBranchValue,
    CompositionCutpoints,
    DecisionFeatures,
    DecisionSnapshot,
    FeatureCutpoint,
    MultipleRegressionAccumulator,
    SampledDecision,
    ScoreConditionedActionValueConfig,
    _new_trajectory,
    aggregate_sampled_actions,
    assign_band,
    branch_action,
    branch_all_legal_actions,
    collect_decision_trajectory,
    composition_labels,
    decision_key,
    decision_time_state,
    discover_candidates,
    freeze_composition_cutpoints,
    observable_table_cards,
    passes_improvement_gate,
    run_score_conditioned_action_values,
    validate_candidates,
)

from shufflemaster_sim.actions import ActionType
from shufflemaster_sim.cards import Card
from shufflemaster_sim.games.casino_blackjack import (
    CasinoBlackjackConfig,
    CasinoBlackjackGame,
)
from shufflemaster_sim.state import HandState
from shufflemaster_sim.strategies.published_casino_strategy import (
    PublishedApproxCasinoStrategy,
)


def test_default_phase_seeds_are_frozen_and_disjoint() -> None:
    assert DEFAULT_DEVELOPMENT_SEEDS == (62, 63, 64, 65, 66)
    assert DEFAULT_VALIDATION_SEEDS == (67, 68, 69, 70, 71)
    assert not set(DEFAULT_DEVELOPMENT_SEEDS).intersection(DEFAULT_VALIDATION_SEEDS)


def test_config_rejects_overlap_changed_weights_and_bad_support() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        ScoreConditionedActionValueConfig(
            development_seeds=(62, 67), validation_seeds=(67, 68)
        )
    with pytest.raises(ValueError, match="frozen"):
        ScoreConditionedActionValueConfig(current_weight=0.9)
    with pytest.raises(ValueError, match="positive"):
        ScoreConditionedActionValueConfig(minimum_total_state_count=0)


def test_cutpoints_use_features_only_and_remain_frozen() -> None:
    features = [feature(-2.0), feature(0.0), feature(2.0)]
    cutpoints = freeze_composition_cutpoints(features, (0.3, 0.7))
    before = cutpoints.as_dict()
    features.append(feature(999.0))
    assert cutpoints.as_dict() == before
    assert not hasattr(cutpoints, "action_outcome")


def test_ace_and_ten_counts_are_separate() -> None:
    counts = CohortCounts.from_cards(
        [card("T", 1), card("J", 2), card("Q", 3), card("K", 4), card("A", 5)]
    )
    assert counts.ten_value == 4
    assert counts.ace == 1


def test_all_nine_ten_ace_cells_assign_independently() -> None:
    cutpoints = uniform_cutpoints()
    cells = set()
    values = {"poor": -1.0, "neutral": 0.0, "rich": 1.0}
    for ten_band, ten_value in values.items():
        for ace_band, ace_value in values.items():
            labels = composition_labels(
                DecisionFeatures(0.0, 0.0, 0.0, ten_value, ace_value), cutpoints
            )
            assert labels[1] == ten_band
            assert labels[2] == ace_band
            cells.add(labels[3])
    assert len(cells) == 9


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, "poor"), (-0.999, "neutral"), (0.999, "neutral"), (1.0, "rich")],
)
def test_low_band_boundaries(value: float, expected: str) -> None:
    assert assign_band(value, FeatureCutpoint(-1.0, 1.0)) == expected


def test_prebet_rack_is_returned_and_table_cards_are_full_weight() -> None:
    config = small_config()
    source, game, _ = _new_trajectory(config, "physical_iid", 67)
    game._pending_discard_rack = [card("2", 90), card("A", 91)]
    source.before_round()
    table = game.create_table(0)
    game.deal_initial_cards(table, source)
    assert game.pending_discard_rack == ()
    assert len(source.ledger.returned) == 1
    assert source.ledger.returned[0].return_draw_index == source.draw_count

    state = decision_time_state(config=config, table=table, source=source)
    assert state.contributions["current_rack"]["card_count"] == 3.0
    assert state.contributions["returned_1_15"]["card_count"] == 1.5


def test_hit_card_enters_table_cohort_and_older_batch_ages_by_draw_count() -> None:
    config = small_config()
    source, game, _ = _new_trajectory(config, "physical_iid", 67)
    table = game.create_table(0)
    table.boxes[0].hands[0].cards = [card("5", 1), card("6", 2)]
    table.dealer.cards = [card("T", 3)]
    source.ledger.returned = [
        ReturnedCohort(
            return_draw_index=source.draw_count - 15,
            counts=CohortCounts.from_cards([card("2", 4)]),
        )
    ]
    first = decision_time_state(config=config, table=table, source=source)
    assert first.contributions["returned_1_15"]["card_count"] == 0.75
    table.boxes[0].hands[0].cards.append(source.draw_card())
    second = decision_time_state(config=config, table=table, source=source)
    assert second.contributions["current_rack"]["card_count"] == 4.0
    assert second.contributions["returned_16_50"]["card_count"] == 0.4


def test_observable_table_cohort_rejects_duplicate_draw_event() -> None:
    game = CasinoBlackjackGame(CasinoBlackjackConfig())
    table = game.create_table(0)
    duplicate = card("8", 1)
    table.boxes[0].hands[0].cards = [duplicate, duplicate]
    table.dealer.cards = [card("6", 2)]
    with pytest.raises(RuntimeError, match="draw event"):
        observable_table_cards(table)


def test_distinct_iid_draws_may_repeat_a_physical_label() -> None:
    game = CasinoBlackjackGame(CasinoBlackjackConfig())
    table = game.create_table(0)
    first = card("8", 1)
    repeated_label = Card(
        rank="8",
        suit="spades",
        physical_id=first.physical_id,
        draw_id=first.draw_id + 1,
    )
    table.boxes[0].hands[0].cards = [first, repeated_label]
    table.dealer.cards = [card("6", 3)]
    assert len(observable_table_cards(table)) == 3


def test_decision_keys_are_stable_for_hard_soft_pair_split_and_legal_sets() -> None:
    game = CasinoBlackjackGame(CasinoBlackjackConfig())
    hard = HandState(0, [card("T", 1), card("6", 2)], 10.0)
    soft = HandState(0, [card("A", 3), card("7", 4)], 10.0)
    pair = HandState(0, [card("T", 5), card("Q", 6)], 10.0)
    split = HandState(1, [card("5", 7), card("6", 8)], 10.0, is_split_hand=True)
    legal = frozenset({ActionType.HIT, ActionType.STAND})
    hard_key = decision_key(
        hand=hard,
        dealer_upcard="6",
        legal_actions=legal,
        baseline_action=ActionType.STAND,
    )
    assert hard_key.hand_kind == "hard"
    assert (
        decision_key(
            hand=soft,
            dealer_upcard="6",
            legal_actions=legal,
            baseline_action=ActionType.STAND,
        ).hand_kind
        == "soft"
    )
    assert (
        decision_key(
            hand=pair,
            dealer_upcard="6",
            legal_actions=legal,
            baseline_action=ActionType.STAND,
        ).pair_value
        == "10"
    )
    assert (
        decision_key(
            hand=split,
            dealer_upcard="6",
            legal_actions=legal,
            baseline_action=ActionType.DOUBLE,
        ).split_hand
        is True
    )
    changed = decision_key(
        hand=hard,
        dealer_upcard="6",
        legal_actions=frozenset({ActionType.HIT}),
        baseline_action=ActionType.HIT,
    )
    assert hard_key.stable_id() != changed.stable_id()
    assert game.config.base_bet == 10.0


def test_action_branches_are_deterministic_and_isolated() -> None:
    snapshot, legal = prepared_snapshot()
    original = deepcopy(snapshot)
    first = branch_all_legal_actions(
        snapshot, legal_actions=legal, strategy=PublishedApproxCasinoStrategy()
    )
    second = branch_all_legal_actions(
        snapshot, legal_actions=legal, strategy=PublishedApproxCasinoStrategy()
    )
    assert first == second
    assert snapshot.table == original.table
    assert snapshot.card_source.draw_count == original.card_source.draw_count
    assert {branch.action for branch in first} == {action.value for action in legal}


def test_natural_sample_contains_every_legal_branch() -> None:
    samples = collect_decision_trajectory(
        small_config(), source_name="one2six", phase="development", seed=62
    )
    assert len(samples) == 1
    sample = samples[0]
    actions = {branch.action for branch in sample.branches}
    assert actions == set(sample.key.legal_actions)
    assert sample.key.baseline_action in actions


def test_double_branch_accounts_for_additional_wager_and_primary_value() -> None:
    snapshot, _ = prepared_snapshot(ranks=("5", "4"))
    result = branch_action(
        snapshot, action=ActionType.DOUBLE, strategy=PublishedApproxCasinoStrategy()
    )
    assert result.initial_box_wager == 10.0
    assert result.additional_action_wager == 10.0
    assert result.total_box_wager == 20.0
    assert result.net_per_initial_wager == result.final_box_net / 10.0
    assert result.net_per_total_wager == result.final_box_net / 20.0


def test_sampled_action_delta_uses_matched_baseline_and_baseline_is_zero() -> None:
    key = synthetic_key()
    sample = SampledDecision(
        source="one2six",
        phase="development",
        seed=62,
        decision_index=0,
        key=key,
        features=feature(0.0),
        branches=(branch("stand", 0.0), branch("hit", 1.0)),
    )
    ten_rows, _ = aggregate_sampled_actions([sample], uniform_cutpoints())
    stand = next(row for row in ten_rows if row["action"] == "stand")
    hit = next(row for row in ten_rows if row["action"] == "hit")
    assert stand["mean_delta_vs_baseline"] == 0.0
    assert hit["mean_delta_vs_baseline"] == 1.0


def test_candidate_gate_rejects_support_seed_ci_and_sign_failures() -> None:
    config = ScoreConditionedActionValueConfig(
        minimum_total_state_count=500,
        minimum_per_seed_state_count=50,
        minimum_seed_sign_count=4,
    )
    valid = candidate_row()
    assert passes_improvement_gate(valid, config, required_seed_count=5)
    for field_name, value in (
        ("sampled_states", 499),
        ("minimum_seed_state_count", 49),
        ("contributing_seeds", 4),
        ("delta_student_t_95_ci", [-0.1, 0.2]),
        ("positive_seed_deltas", 3),
    ):
        changed = dict(valid)
        changed[field_name] = value
        assert not passes_improvement_gate(changed, config, required_seed_count=5)


def test_candidate_classification_distinguishes_generic_and_one2six_specific() -> None:
    config = gate_config()
    one = aggregate_candidate_row("one2six", delta=0.2)
    iid_same = aggregate_candidate_row("physical_iid", delta=0.2)
    generic = discover_candidates(
        config,
        [one, iid_same],
        per_seed_candidate_rows(0.2, 0.2, phase="development"),
    )
    assert "generic_baseline_correction" in generic[0]["candidate_classes"]

    iid_zero = aggregate_candidate_row("physical_iid", delta=0.0, positive=False)
    specific = discover_candidates(
        config,
        [one, iid_zero],
        per_seed_candidate_rows(0.2, 0.0, phase="development"),
    )
    assert "one2six_composition_candidate" in specific[0]["candidate_classes"]


def test_candidate_classifies_loss_reduction_and_possible_edge_creation() -> None:
    config = gate_config()
    loss = aggregate_candidate_row(
        "one2six",
        delta=0.2,
        action_value=-0.2,
        baseline_value=-0.4,
        family="low_band",
        composition_state="rich",
    )
    iid = aggregate_candidate_row(
        "physical_iid",
        delta=0.0,
        positive=False,
        family="low_band",
        composition_state="rich",
    )
    candidates = discover_candidates(
        config,
        [loss, iid],
        per_seed_candidate_rows(
            0.2,
            0.0,
            phase="development",
            family="low_band",
            composition_state="rich",
        ),
    )
    assert "loss_reduction_candidate" in candidates[0]["candidate_classes"]

    edge = aggregate_candidate_row(
        "one2six", delta=0.4, action_value=0.2, baseline_value=-0.2
    )
    edge["action_value_seed_ci"] = [0.05, 0.35]
    candidates = discover_candidates(
        config,
        [edge, aggregate_candidate_row("physical_iid", 0.0, positive=False)],
        per_seed_candidate_rows(0.4, 0.0, phase="development"),
    )
    assert "possible_edge_creation_candidate" in candidates[0]["candidate_classes"]


def test_heldout_validation_cannot_add_candidates_and_applies_strict_labels() -> None:
    config = gate_config()
    development_one = aggregate_candidate_row("one2six", 0.2)
    development_iid = aggregate_candidate_row("physical_iid", 0.0, positive=False)
    candidates = discover_candidates(
        config,
        [development_one, development_iid],
        per_seed_candidate_rows(0.2, 0.0, phase="development"),
    )
    frozen = deepcopy(candidates)
    validation_one = aggregate_candidate_row("one2six", 0.25, phase="validation")
    validation_iid = aggregate_candidate_row(
        "physical_iid", 0.0, phase="validation", positive=False
    )
    results, validated = validate_candidates(
        config,
        candidates,
        [validation_one, validation_iid],
        per_seed_candidate_rows(0.25, 0.0, phase="validation"),
    )
    assert candidates == frozen
    assert len(results) == 1
    assert validated[0]["validation_labels"] == ["validated_one2six_deviation"]

    validation_one["delta_student_t_95_ci"] = [-0.1, 0.3]
    _, failed = validate_candidates(
        config,
        candidates,
        [validation_one, validation_iid],
        per_seed_candidate_rows(0.25, 0.0, phase="validation"),
    )
    assert failed == []


def test_multivariate_regression_recovers_separate_low_ten_and_ace_coefficients() -> (
    None
):
    regression = MultipleRegressionAccumulator()
    for low in (-2.0, -1.0, 1.0, 2.0):
        for ten in (-1.5, 0.5):
            for ace in (-0.75, 1.25):
                regression.add(low, ten, ace, 1.0 + 2 * low + 3 * ten + 4 * ace)
    coefficients = regression.coefficients()
    assert coefficients is not None
    assert coefficients == pytest.approx((1.0, 2.0, 3.0, 4.0))


def test_small_experiment_freezes_candidates_and_exports_no_hidden_state(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "action-values"
    summary = run_score_conditioned_action_values(
        ScoreConditionedActionValueConfig(
            development_seeds=(62,),
            validation_seeds=(67,),
            decision_states_per_seed=20,
            burn_in_rounds=2,
            minimum_total_state_count=1,
            minimum_per_seed_state_count=1,
            minimum_seed_sign_count=1,
            output_dir=output_dir,
        )
    )
    assert summary["hidden_state_exported"] is False
    assert sum(row["decision_snapshots"] for row in summary["coverage"]) == 80
    required = {
        "summary.json",
        "composition_cutpoints.json",
        "development_candidates.json",
        "validation_candidate_results.csv",
        "validated_candidates.json",
        "decision_state_frequency.csv",
    }
    assert required.issubset(path.name for path in output_dir.iterdir())
    candidate_header = (output_dir / "validation_candidate_results.csv").read_text(
        encoding="utf-8"
    )
    assert candidate_header.startswith("candidate_id,")
    exported = "".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.iterdir()
        if path.suffix in {".json", ".csv", ".md"}
    ).lower()
    assert not any(term in exported for term in PRIVATE_TERMS)


def small_config() -> ScoreConditionedActionValueConfig:
    return ScoreConditionedActionValueConfig(
        development_seeds=(62,),
        validation_seeds=(67,),
        decision_states_per_seed=1,
        burn_in_rounds=0,
        minimum_total_state_count=1,
        minimum_per_seed_state_count=1,
        minimum_seed_sign_count=1,
    )


def card(rank: str, index: int) -> Card:
    return Card(
        rank=rank,
        suit="spades",
        physical_id=f"test:{index}",
        draw_id=1_000 + index,
    )


def feature(value: float) -> DecisionFeatures:
    return DecisionFeatures(value, value, value, value, value)


def uniform_cutpoints() -> CompositionCutpoints:
    cutpoint = FeatureCutpoint(-0.5, 0.5)
    return CompositionCutpoints(cutpoint, cutpoint, cutpoint)


def prepared_snapshot(
    ranks: tuple[str, str] = ("T", "6"),
) -> tuple[DecisionSnapshot, frozenset[ActionType]]:
    game = CasinoBlackjackGame(CasinoBlackjackConfig(base_bet=10.0, box_count=1))
    table = game.create_table(0)
    table.boxes[0].hands[0].cards = [card(ranks[0], 1), card(ranks[1], 2)]
    table.dealer.cards = [card("6", 3)]
    source = LedgerCardSource(
        _make_source("physical_iid", 6, 99), ObservableCohortLedger()
    )
    hand = table.boxes[0].hands[0]
    legal = game.legal_actions(table=table, box=table.boxes[0], hand=hand)
    return DecisionSnapshot(game, table, source, 1, 0), legal


def synthetic_key():
    snapshot, legal = prepared_snapshot()
    hand = snapshot.table.boxes[0].hands[0]
    return decision_key(
        hand=hand,
        dealer_upcard="6",
        legal_actions=legal,
        baseline_action=ActionType.STAND,
    )


def branch(action: str, value: float) -> ActionBranchValue:
    return ActionBranchValue(action, value * 10, 10, 0, 10, value, value)


def candidate_row() -> dict[str, object]:
    return {
        "action": "hit",
        "baseline_action": "stand",
        "sampled_states": 500,
        "contributing_seeds": 5,
        "minimum_seed_state_count": 50,
        "mean_delta_vs_baseline": 0.1,
        "delta_student_t_95_ci": [0.01, 0.2],
        "positive_seed_deltas": 4,
    }


def gate_config() -> ScoreConditionedActionValueConfig:
    return ScoreConditionedActionValueConfig(
        minimum_total_state_count=5,
        minimum_per_seed_state_count=1,
        minimum_seed_sign_count=4,
    )


def aggregate_candidate_row(
    source: str,
    delta: float,
    *,
    phase: str = "development",
    action_value: float | None = None,
    baseline_value: float = -0.2,
    positive: bool = True,
    family: str = "ten_ace",
    composition_state: str = "ten_rich__ace_poor",
) -> dict[str, object]:
    value = baseline_value + delta if action_value is None else action_value
    return {
        "source": source,
        "phase": phase,
        "family": family,
        "decision_id": "decision",
        "composition_state": composition_state,
        "action": "hit",
        "baseline_action": "stand",
        "hand_kind": "hard",
        "hand_total": 16,
        "pair_value": None,
        "dealer_upcard": "10",
        "card_count": 2,
        "split_hand": False,
        "split_aces": False,
        "legal_actions": "hit|stand",
        "sampled_states": 5,
        "contributing_seeds": 5,
        "minimum_seed_state_count": 1,
        "mean_delta_vs_baseline": delta,
        "delta_student_t_95_ci": [0.1, 0.3] if positive else [-0.1, 0.1],
        "positive_seed_deltas": 5 if positive else 2,
        "mean_action_value": value,
        "action_value_seed_ci": [-0.3, 0.3],
        "positive_seed_action_values": 3,
        "mean_baseline_value": baseline_value,
        "baseline_value_seed_ci": [-0.3, 0.0],
    }


def per_seed_candidate_rows(
    one_delta: float,
    iid_delta: float,
    *,
    phase: str,
    family: str = "ten_ace",
    composition_state: str = "ten_rich__ace_poor",
) -> list[dict[str, object]]:
    rows = []
    seeds = (
        DEFAULT_DEVELOPMENT_SEEDS
        if phase == "development"
        else DEFAULT_VALIDATION_SEEDS
    )
    for seed in seeds:
        for source, delta in (("one2six", one_delta), ("physical_iid", iid_delta)):
            rows.append(
                {
                    "source": source,
                    "phase": phase,
                    "seed": seed,
                    "family": family,
                    "decision_id": "decision",
                    "composition_state": composition_state,
                    "action": "hit",
                    "mean_delta_vs_baseline": delta,
                }
            )
    return rows
