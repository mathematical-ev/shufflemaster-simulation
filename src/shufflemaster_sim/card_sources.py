"""Card source abstractions for deterministic and stochastic draws."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from random import Random
from typing import Final, Protocol, TypeAlias

from shufflemaster_sim.cards import RANKS, SUITS, Card, Rank, Suit

CardSpec: TypeAlias = Card | tuple[Rank, Suit]
ALLOWED_FINITE_SHOE_DECK_COUNTS: Final[frozenset[int]] = frozenset({1, 4, 6, 8})
MANUAL_SHOE_MIN_ROUND_CARDS: Final[int] = 20


class CardSource(Protocol):
    """Protocol for anything that can provide dealt card events."""

    def before_round(self) -> None:
        """Run any source-specific round-boundary maintenance."""

    def draw_card(self) -> Card:
        """Draw one card event."""

    def accept_discards(self, cards: Sequence[Card]) -> None:
        """Accept cards that have left play."""


class ScriptedCardSource:
    """Deterministic source that returns a fixed sequence of cards."""

    def __init__(self, cards: Iterable[CardSpec], *, start_draw_id: int = 0) -> None:
        self._cards: Iterator[CardSpec] = iter(cards)
        self._next_draw_id = start_draw_id
        self.accepted_discards: list[Card] = []
        self.accepted_discard_batches: list[list[Card]] = []

    def before_round(self) -> None:
        """No-op round hook for scripted cards."""

    def draw_card(self) -> Card:
        """Return the next scripted card with a unique draw id."""
        try:
            card_spec = next(self._cards)
        except StopIteration as exc:
            raise RuntimeError("ScriptedCardSource is exhausted.") from exc

        draw_id = self._consume_draw_id()
        if isinstance(card_spec, Card):
            return replace(card_spec, draw_id=draw_id)

        rank, suit = card_spec
        return Card(
            rank=rank,
            suit=suit,
            physical_id=f"scripted:{draw_id}",
            draw_id=draw_id,
        )

    def _consume_draw_id(self) -> int:
        draw_id = self._next_draw_id
        self._next_draw_id += 1
        return draw_id

    def accept_discards(self, cards: Sequence[Card]) -> None:
        """Record discards without changing the scripted draw order."""
        self.accepted_discard_batches.append(list(cards))
        self.accepted_discards.extend(cards)


class IidRandomCardSource:
    """Independent random card source for baseline mechanics tests."""

    def __init__(
        self,
        *,
        rng: Random | None = None,
        seed: int | str | bytes | bytearray | None = None,
        start_draw_id: int = 0,
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")

        self._rng = rng if rng is not None else Random(seed)
        self._next_draw_id = start_draw_id

    def before_round(self) -> None:
        """No-op round hook for IID draws."""

    def draw_card(self) -> Card:
        """Draw a random rank/suit event independently of previous draws."""
        draw_id = self._consume_draw_id()
        card = Card(
            rank=self._rng.choice(RANKS),
            suit=self._rng.choice(SUITS),
            physical_id=f"iid:{draw_id}",
            draw_id=draw_id,
        )
        return card

    def accept_discards(self, cards: Sequence[Card]) -> None:
        """Accept discarded cards.

        IID draws have no memory, so discards do not affect future cards.
        """
        _ = cards

    def _consume_draw_id(self) -> int:
        draw_id = self._next_draw_id
        self._next_draw_id += 1
        return draw_id


class FiniteShoeCardSource:
    """Finite shuffled shoe that draws physical cards without replacement."""

    def __init__(
        self,
        *,
        deck_count: int = 6,
        rng: Random | None = None,
        seed: int | str | bytes | bytearray | None = None,
        reshuffle_when_empty: bool = True,
        start_draw_id: int = 0,
    ) -> None:
        if deck_count not in ALLOWED_FINITE_SHOE_DECK_COUNTS:
            allowed = ", ".join(
                str(count) for count in sorted(ALLOWED_FINITE_SHOE_DECK_COUNTS)
            )
            raise ValueError(f"deck_count must be one of: {allowed}.")
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")

        self.deck_count = deck_count
        self.reshuffle_when_empty = reshuffle_when_empty
        self._rng = rng if rng is not None else Random(seed)
        self._next_draw_id = start_draw_id
        self._shoe = self._build_physical_cards(deck_count)
        self._rng.shuffle(self._shoe)
        self.discard_tray: list[Card] = []
        self.accepted_discards: list[Card] = []
        self.accepted_discard_batches: list[list[Card]] = []

    def before_round(self) -> None:
        """No-op round hook for the generic exhaustion-based finite source."""

    @property
    def cards_remaining(self) -> int:
        """Return the number of drawable cards currently in the shoe."""
        return len(self._shoe)

    @property
    def discard_count(self) -> int:
        """Return the number of accepted cards waiting for reshuffle."""
        return len(self.discard_tray)

    @property
    def shoe_cards(self) -> tuple[Card, ...]:
        """Return a read-only snapshot of the current shoe."""
        return tuple(self._shoe)

    def draw_card(self) -> Card:
        """Draw one physical card from the finite shoe."""
        if not self._shoe:
            self._reshuffle_if_possible()

        physical_card = self._shoe.pop()
        return replace(physical_card, draw_id=self._consume_draw_id())

    def accept_discards(self, cards: Sequence[Card]) -> None:
        """Accept ordered discards into the tray without making them drawable."""
        batch = list(cards)
        self.accepted_discard_batches.append(batch)
        self.accepted_discards.extend(batch)
        self.discard_tray.extend(batch)

    def _reshuffle_if_possible(self) -> None:
        if self.reshuffle_when_empty and self.discard_tray:
            self._shoe = list(self.discard_tray)
            self.discard_tray.clear()
            self._rng.shuffle(self._shoe)
            return
        raise RuntimeError(
            "FiniteShoeCardSource is empty and has no discards to reshuffle."
        )

    def _consume_draw_id(self) -> int:
        draw_id = self._next_draw_id
        self._next_draw_id += 1
        return draw_id

    def _build_physical_cards(self, deck_count: int) -> list[Card]:
        return [
            Card(
                rank=rank,
                suit=suit,
                physical_id=f"deck-{deck_index}:{rank}:{suit}",
                draw_id=-1,
            )
            for deck_index in range(deck_count)
            for suit in SUITS
            for rank in RANKS
        ]


class ManualShoeCardSource:
    """Normal manual shoe with cut-card penetration and round-boundary shuffles."""

    def __init__(
        self,
        *,
        deck_count: int = 8,
        cut_card_penetration: float = 0.75,
        rng: Random | None = None,
        seed: int | str | bytes | bytearray | None = None,
        start_draw_id: int = 0,
    ) -> None:
        if deck_count not in ALLOWED_FINITE_SHOE_DECK_COUNTS:
            allowed = ", ".join(
                str(count) for count in sorted(ALLOWED_FINITE_SHOE_DECK_COUNTS)
            )
            raise ValueError(f"deck_count must be one of: {allowed}.")
        if not 0 < cut_card_penetration <= 1:
            raise ValueError("cut_card_penetration must be in (0, 1].")
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")

        self.deck_count = deck_count
        self.cut_card_penetration = cut_card_penetration
        self._rng = rng if rng is not None else Random(seed)
        self._next_draw_id = start_draw_id
        self._shoe = self._build_physical_cards(deck_count)
        self._rng.shuffle(self._shoe)
        self.discard_tray: list[Card] = []
        self.accepted_discards: list[Card] = []
        self.accepted_discard_batches: list[list[Card]] = []
        self.cards_dealt_since_shuffle = 0
        self.reshuffle_pending = False
        self.shuffle_count = 0

    @property
    def cards_remaining(self) -> int:
        """Return the number of drawable cards currently in the shoe."""
        return len(self._shoe)

    @property
    def discard_count(self) -> int:
        """Return the number of accepted cards waiting in the discard tray."""
        return len(self.discard_tray)

    @property
    def shoe_cards(self) -> tuple[Card, ...]:
        """Return a read-only snapshot of the current shoe."""
        return tuple(self._shoe)

    @property
    def cut_card_position(self) -> int:
        """Return the dealt-card count that marks the cut-card threshold."""
        return max(1, int(self.deck_count * 52 * self.cut_card_penetration))

    def before_round(self) -> None:
        """Shuffle before a new round when the cut-card threshold was crossed."""
        if self.reshuffle_pending or (
            self.cards_remaining < MANUAL_SHOE_MIN_ROUND_CARDS and self.discard_tray
        ):
            self.shuffle()

    def draw_card(self) -> Card:
        """Draw one physical card without replacement."""
        if not self._shoe:
            self.shuffle()
        if not self._shoe:
            raise RuntimeError("ManualShoeCardSource has no cards available to draw.")

        physical_card = self._shoe.pop()
        self.cards_dealt_since_shuffle += 1
        if self.cards_dealt_since_shuffle >= self.cut_card_position:
            self.reshuffle_pending = True
        return replace(physical_card, draw_id=self._consume_draw_id())

    def accept_discards(self, cards: Sequence[Card]) -> None:
        """Accept ordered discards without making them drawable."""
        batch = list(cards)
        self.accepted_discard_batches.append(batch)
        self.accepted_discards.extend(batch)
        self.discard_tray.extend(batch)

    def shuffle(self) -> None:
        """Shuffle remaining shoe cards and discard tray into a fresh shoe."""
        cards_to_shuffle = [*self._shoe, *self.discard_tray]
        if not cards_to_shuffle:
            raise RuntimeError("ManualShoeCardSource has no cards to shuffle.")

        self._shoe = cards_to_shuffle
        self.discard_tray.clear()
        self._rng.shuffle(self._shoe)
        self.cards_dealt_since_shuffle = 0
        self.reshuffle_pending = False
        self.shuffle_count += 1

    def _consume_draw_id(self) -> int:
        draw_id = self._next_draw_id
        self._next_draw_id += 1
        return draw_id

    def _build_physical_cards(self, deck_count: int) -> list[Card]:
        return [
            Card(
                rank=rank,
                suit=suit,
                physical_id=f"deck-{deck_index}:{rank}:{suit}",
                draw_id=-1,
            )
            for deck_index in range(deck_count)
            for suit in SUITS
            for rank in RANKS
        ]
