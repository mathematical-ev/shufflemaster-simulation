# SPDX-License-Identifier: GPL-3.0-or-later

from shufflemaster_sim.cards import Card, blackjack_value


def test_ace_returns_eleven_as_raw_value() -> None:
    assert blackjack_value("A") == 11
    assert (
        blackjack_value(Card(rank="A", suit="spades", physical_id="test:0", draw_id=0))
        == 11
    )


def test_ten_value_ranks_return_ten() -> None:
    assert {rank: blackjack_value(rank) for rank in ("T", "J", "Q", "K")} == {
        "T": 10,
        "J": 10,
        "Q": 10,
        "K": 10,
    }


def test_numeric_ranks_return_numeric_value() -> None:
    assert {rank: blackjack_value(rank) for rank in ("2", "3", "4", "5", "6")} == {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
    }
    assert {rank: blackjack_value(rank) for rank in ("7", "8", "9")} == {
        "7": 7,
        "8": 8,
        "9": 9,
    }


def test_package_imports_work_from_src_layout() -> None:
    import shufflemaster_sim

    assert shufflemaster_sim.Card is Card
    assert shufflemaster_sim.blackjack_value("K") == 10
