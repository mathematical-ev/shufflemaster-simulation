from collections import Counter, deque

import pytest

from shufflemaster_sim.card_sources import (
    NoEligibleSlotError,
    One2SixCardSource,
    One2SixConfig,
)
from shufflemaster_sim.cards import Card
from shufflemaster_sim.simulation import SimulationConfig, run_star_blackjack_baseline


def test_default_one2six_config_and_initial_load() -> None:
    source = One2SixCardSource(seed=42)

    assert source.config.deck_count == 6
    assert source.config.carousel_slot_count == 38
    assert source.config.output_buffer_target == 18
    assert source.config.refill_threshold == 8
    assert source.config.min_cards_for_ejection == 7
    assert source.cards_total_known == 312
    assert source.output_buffer_size >= source.config.output_buffer_target
    assert source.cards_in_carousel + source.output_buffer_size == 312


def test_initial_physical_ids_are_unique_with_six_copies_per_rank_suit() -> None:
    source = One2SixCardSource(seed=42)
    cards = [
        *source.output_buffer_cards,
        *(card for shelf in source._carousel for card in shelf),
    ]

    assert len({card.physical_id for card in cards}) == 312
    assert set(Counter((card.rank, card.suit) for card in cards).values()) == {6}


def test_draw_preserves_physical_identity_and_assigns_new_draw_id() -> None:
    source = One2SixCardSource(seed=42)
    expected_physical_id = source.output_buffer_cards[0].physical_id

    drawn = source.draw_card()

    assert drawn.physical_id == expected_physical_id
    assert drawn.draw_id == 0


def test_accepted_discards_preserve_identity_and_batch_order() -> None:
    source = One2SixCardSource(seed=42)
    discards = [source.draw_card(), source.draw_card()]

    source.accept_discards(discards)

    assert source.accepted_discard_batches == [discards]
    assert [card.physical_id for card in source.accepted_discards] == [
        card.physical_id for card in discards
    ]


def test_accepted_cards_are_fed_in_order_for_fixed_slot_rule() -> None:
    config = One2SixConfig(
        deck_count=1,
        carousel_slot_count=38,
        slot_capacity=60,
        output_buffer_target=18,
        insertion_rule="fixed_slot_for_testing",
    )
    source = One2SixCardSource(config=config, seed=42)
    accepted = [source.draw_card(), source.draw_card(), source.draw_card()]

    source.accept_discards(accepted)

    entered_shelf = [
        record["physical_id"]
        for record in source.telemetry_records()
        if record["event_type"] == "entered_carousel_shelf"
    ]
    assert entered_shelf[-3:] == [card.physical_id for card in accepted]


def test_lifo_whole_shelf_ejection_appends_to_back_of_output_buffer() -> None:
    config = One2SixConfig(
        deck_count=1,
        carousel_slot_count=1,
        slot_capacity=60,
        output_buffer_target=1,
        refill_threshold=0,
        min_cards_for_ejection=3,
        insertion_rule="fixed_slot_for_testing",
        output_selection_rule="fixed_slot_for_testing",
        strict_invariants=False,
    )
    source = One2SixCardSource(config=config, seed=42)
    source._output_buffer = deque(
        [Card(rank="2", suit="clubs", physical_id="sentinel", draw_id=-1)]
    )
    source._carousel = [
        [
            Card(rank="A", suit="spades", physical_id="A", draw_id=-1),
            Card(rank="2", suit="spades", physical_id="B", draw_id=-1),
            Card(rank="3", suit="spades", physical_id="C", draw_id=-1),
        ]
    ]

    source._eject_shelf(0)

    assert source._carousel[0] == []
    assert [card.physical_id for card in source.output_buffer_cards] == [
        "sentinel",
        "C",
        "B",
        "A",
    ]


def test_buffer_refill_triggers_when_threshold_is_reached() -> None:
    source = One2SixCardSource(seed=42)
    initial_ejections = source.ejection_count

    for _ in range(source.output_buffer_size - source.config.refill_threshold):
        source.draw_card()

    assert source.ejection_count > initial_ejections
    assert source.output_buffer_size >= source.config.refill_threshold


def test_fallback_chooses_fullest_occupied_slot_and_records_event() -> None:
    config = One2SixConfig(
        deck_count=1,
        carousel_slot_count=4,
        slot_capacity=20,
        output_buffer_target=18,
        refill_threshold=8,
        min_cards_for_ejection=100,
    )

    source = One2SixCardSource(config=config, seed=42)

    assert source.fallback_ejection_count > 0
    assert any(record["used_fallback"] for record in source.ejection_records())


def test_strict_no_eligible_slot_can_raise() -> None:
    config = One2SixConfig(
        deck_count=1,
        carousel_slot_count=4,
        slot_capacity=20,
        output_buffer_target=18,
        refill_threshold=8,
        min_cards_for_ejection=100,
        fallback_when_no_eligible_slot="raise",
    )

    with pytest.raises(NoEligibleSlotError):
        One2SixCardSource(config=config, seed=42)


def test_invariants_pass_after_draws_and_discards() -> None:
    source = One2SixCardSource(seed=42)
    external_cards = [source.draw_card() for _ in range(20)]
    source.accept_discards(external_cards[:10])

    source.assert_invariants(external_cards=external_cards[10:])


def test_invariants_fail_when_duplicate_physical_id_is_inserted() -> None:
    source = One2SixCardSource(seed=42)
    duplicate = source.output_buffer_cards[0]
    source._output_buffer.append(duplicate)

    with pytest.raises(AssertionError, match="Duplicate physical id"):
        source.assert_invariants()


def test_same_seed_reproduces_initial_draw_sequence() -> None:
    first = One2SixCardSource(seed=42)
    second = One2SixCardSource(seed=42)

    assert [first.draw_card().physical_id for _ in range(30)] == [
        second.draw_card().physical_id for _ in range(30)
    ]


def test_different_seed_may_change_initial_draw_sequence() -> None:
    first = One2SixCardSource(seed=1)
    second = One2SixCardSource(seed=2)

    assert [first.draw_card().physical_id for _ in range(30)] != [
        second.draw_card().physical_id for _ in range(30)
    ]


def test_star_baseline_runs_with_one2six_source() -> None:
    result = run_star_blackjack_baseline(
        SimulationConfig(rounds=25, seed=42, card_source="one2six")
    )

    assert result.rounds_played == 25
    assert result.output_buffer_size is not None
    assert result.ejection_count is not None
    assert result.fallback_ejection_count is not None


def test_accepted_discards_do_not_enter_output_buffer_directly() -> None:
    source = One2SixCardSource(seed=42)
    discards = [source.draw_card(), source.draw_card()]
    source.accept_discards(discards)

    output_ids = {card.physical_id for card in source.output_buffer_cards}

    assert not output_ids.intersection(card.physical_id for card in discards)
