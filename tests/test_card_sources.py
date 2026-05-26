import pytest

from shufflemaster_sim.card_sources import IidRandomCardSource, ScriptedCardSource
from shufflemaster_sim.cards import Card


def test_scripted_card_source_returns_cards_in_order() -> None:
    source = ScriptedCardSource([("A", "spades"), ("5", "hearts"), ("K", "clubs")])

    assert source.draw_card() == Card(
        rank="A",
        suit="spades",
        physical_id="scripted:0",
        draw_id=0,
    )
    assert source.draw_card() == Card(
        rank="5",
        suit="hearts",
        physical_id="scripted:1",
        draw_id=1,
    )
    assert source.draw_card() == Card(
        rank="K",
        suit="clubs",
        physical_id="scripted:2",
        draw_id=2,
    )


def test_scripted_card_source_reassigns_unique_draw_ids() -> None:
    source = ScriptedCardSource(
        [
            Card(rank="A", suit="spades", physical_id="ace:1", draw_id=99),
            Card(rank="A", suit="spades", physical_id="ace:2", draw_id=99),
        ],
        start_draw_id=10,
    )

    assert source.draw_card() == Card(
        rank="A",
        suit="spades",
        physical_id="ace:1",
        draw_id=10,
    )
    assert source.draw_card() == Card(
        rank="A",
        suit="spades",
        physical_id="ace:2",
        draw_id=11,
    )


def test_scripted_card_source_raises_clear_error_when_exhausted() -> None:
    source = ScriptedCardSource([])

    with pytest.raises(RuntimeError, match="ScriptedCardSource is exhausted"):
        source.draw_card()


def test_iid_random_card_source_is_reproducible_when_seeded() -> None:
    first = IidRandomCardSource(seed=12345)
    second = IidRandomCardSource(seed=12345)

    assert [first.draw_card() for _ in range(12)] == [
        second.draw_card() for _ in range(12)
    ]


def test_iid_random_card_source_assigns_unique_draw_ids() -> None:
    source = IidRandomCardSource(seed=12345)

    draw_ids = [source.draw_card().draw_id for _ in range(100)]

    assert draw_ids == list(range(100))
    assert len(set(draw_ids)) == len(draw_ids)


def test_iid_random_card_source_assigns_new_physical_identity_per_draw() -> None:
    source = IidRandomCardSource(seed=12345)

    physical_ids = [source.draw_card().physical_id for _ in range(100)]

    assert len(set(physical_ids)) == len(physical_ids)
