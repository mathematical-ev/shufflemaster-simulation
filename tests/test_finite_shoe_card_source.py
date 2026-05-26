from collections import Counter

import pytest

from shufflemaster_sim.card_sources import FiniteShoeCardSource


def physical_ids(source: FiniteShoeCardSource) -> list[str]:
    return [card.physical_id for card in source.shoe_cards]


def test_six_deck_finite_shoe_has_unique_physical_identities() -> None:
    source = FiniteShoeCardSource(deck_count=6, seed=42)

    ids = physical_ids(source)

    assert len(ids) == 312
    assert len(ids) == len(set(ids))


def test_six_deck_finite_shoe_has_six_copies_of_each_rank_suit() -> None:
    source = FiniteShoeCardSource(deck_count=6, seed=42)

    counts = Counter((card.rank, card.suit) for card in source.shoe_cards)

    assert len(counts) == 52
    assert set(counts.values()) == {6}


def test_drawing_preserves_physical_identity_and_assigns_draw_identity() -> None:
    source = FiniteShoeCardSource(deck_count=1, seed=42)
    next_physical_card = source.shoe_cards[-1]

    drawn = source.draw_card()

    assert drawn.physical_id == next_physical_card.physical_id
    assert drawn.draw_id == 0
    assert next_physical_card.draw_id == -1


def test_accepting_discards_preserves_order_and_identity() -> None:
    source = FiniteShoeCardSource(deck_count=1, seed=42)
    drawn_cards = [source.draw_card() for _ in range(3)]

    source.accept_discards(drawn_cards)

    assert source.discard_tray == drawn_cards
    assert source.accepted_discard_batches == [drawn_cards]


def test_reshuffle_returns_same_physical_identities_to_shoe() -> None:
    source = FiniteShoeCardSource(deck_count=1, seed=42)
    first_pass = [source.draw_card() for _ in range(52)]
    first_pass_ids = {card.physical_id for card in first_pass}
    source.accept_discards(first_pass)

    second_pass = [source.draw_card() for _ in range(52)]

    assert {card.physical_id for card in second_pass} == first_pass_ids
    assert {card.draw_id for card in second_pass} == set(range(52, 104))


def test_one_deck_shoe_draws_exactly_fifty_two_cards_before_exhaustion() -> None:
    source = FiniteShoeCardSource(
        deck_count=1,
        seed=42,
        reshuffle_when_empty=False,
    )
    drawn = [source.draw_card() for _ in range(52)]

    assert len({card.physical_id for card in drawn}) == 52
    with pytest.raises(RuntimeError, match="empty and has no discards"):
        source.draw_card()


def test_same_seed_reproduces_finite_shoe_draw_order() -> None:
    first = FiniteShoeCardSource(deck_count=1, seed=42)
    second = FiniteShoeCardSource(deck_count=1, seed=42)

    assert [first.draw_card().physical_id for _ in range(52)] == [
        second.draw_card().physical_id for _ in range(52)
    ]


def test_different_seeds_can_change_finite_shoe_draw_order() -> None:
    first = FiniteShoeCardSource(deck_count=1, seed=1)
    second = FiniteShoeCardSource(deck_count=1, seed=2)

    assert [first.draw_card().physical_id for _ in range(52)] != [
        second.draw_card().physical_id for _ in range(52)
    ]


def test_empty_shoe_with_accepted_discards_reshuffles_when_enabled() -> None:
    source = FiniteShoeCardSource(deck_count=1, seed=42)
    first_pass = [source.draw_card() for _ in range(52)]
    source.accept_discards(first_pass)

    redrawn = source.draw_card()

    assert redrawn.physical_id in {card.physical_id for card in first_pass}
    assert redrawn.draw_id == 52


def test_accepted_discards_are_not_drawable_until_reshuffle() -> None:
    source = FiniteShoeCardSource(deck_count=1, seed=42)
    discarded = source.draw_card()
    source.accept_discards([discarded])

    remaining_before_reshuffle = [source.draw_card() for _ in range(51)]

    assert discarded.physical_id not in {
        card.physical_id for card in remaining_before_reshuffle
    }
    assert source.draw_card().physical_id == discarded.physical_id
