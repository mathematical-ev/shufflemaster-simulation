from collections import Counter

from shufflemaster_sim.card_sources import ManualShoeCardSource


def test_default_manual_shoe_uses_eight_decks() -> None:
    source = ManualShoeCardSource(seed=42)

    assert source.deck_count == 8
    assert source.cards_remaining == 416


def test_six_deck_manual_shoe_can_be_configured() -> None:
    source = ManualShoeCardSource(deck_count=6, seed=42)

    assert source.deck_count == 6
    assert source.cards_remaining == 312


def test_eight_deck_manual_shoe_has_unique_physical_identities() -> None:
    source = ManualShoeCardSource(seed=42)

    physical_ids = [card.physical_id for card in source.shoe_cards]

    assert len(physical_ids) == 416
    assert len(physical_ids) == len(set(physical_ids))


def test_no_physical_card_appears_twice_before_reshuffle() -> None:
    source = ManualShoeCardSource(deck_count=1, seed=42)

    drawn = [source.draw_card() for _ in range(52)]

    assert len({card.physical_id for card in drawn}) == 52
    assert source.reshuffle_pending
    assert source.shuffle_count == 0


def test_accepted_discards_preserve_order_and_identity() -> None:
    source = ManualShoeCardSource(deck_count=1, seed=42)
    discards = [source.draw_card() for _ in range(3)]

    source.accept_discards(discards)

    assert source.discard_tray == discards
    assert source.accepted_discard_batches == [discards]


def test_accepted_discards_are_not_drawable_before_reshuffle() -> None:
    source = ManualShoeCardSource(deck_count=1, seed=42)
    discarded = source.draw_card()
    source.accept_discards([discarded])

    remaining_before_shuffle = [source.draw_card() for _ in range(51)]

    assert discarded.physical_id not in {
        card.physical_id for card in remaining_before_shuffle
    }


def test_cut_card_penetration_sets_reshuffle_pending() -> None:
    source = ManualShoeCardSource(
        deck_count=1,
        cut_card_penetration=0.10,
        seed=42,
    )

    for _ in range(source.cut_card_position):
        source.draw_card()

    assert source.reshuffle_pending
    assert source.cards_dealt_since_shuffle == source.cut_card_position


def test_crossing_penetration_does_not_shuffle_mid_round() -> None:
    source = ManualShoeCardSource(
        deck_count=1,
        cut_card_penetration=0.10,
        seed=42,
    )

    for _ in range(source.cut_card_position + 2):
        source.draw_card()

    assert source.reshuffle_pending
    assert source.shuffle_count == 0


def test_before_round_reshuffles_when_pending() -> None:
    source = ManualShoeCardSource(
        deck_count=1,
        cut_card_penetration=0.10,
        seed=42,
    )
    drawn = [source.draw_card() for _ in range(source.cut_card_position)]
    source.accept_discards(drawn[:2])
    expected_physical_ids = {
        *(card.physical_id for card in source.shoe_cards),
        *(card.physical_id for card in source.discard_tray),
    }

    source.before_round()

    assert source.shuffle_count == 1
    assert not source.reshuffle_pending
    assert source.cards_dealt_since_shuffle == 0
    assert source.discard_tray == []
    assert {card.physical_id for card in source.shoe_cards} == expected_physical_ids


def test_same_seed_reproduces_manual_shoe_order() -> None:
    first = ManualShoeCardSource(deck_count=1, seed=42)
    second = ManualShoeCardSource(deck_count=1, seed=42)

    assert [first.draw_card().physical_id for _ in range(52)] == [
        second.draw_card().physical_id for _ in range(52)
    ]


def test_different_seed_may_change_manual_shoe_order() -> None:
    first = ManualShoeCardSource(deck_count=1, seed=1)
    second = ManualShoeCardSource(deck_count=1, seed=2)

    assert [first.draw_card().physical_id for _ in range(52)] != [
        second.draw_card().physical_id for _ in range(52)
    ]


def test_eight_deck_manual_shoe_has_eight_copies_of_each_rank_suit() -> None:
    source = ManualShoeCardSource(seed=42)

    counts = Counter((card.rank, card.suit) for card in source.shoe_cards)

    assert len(counts) == 52
    assert set(counts.values()) == {8}
