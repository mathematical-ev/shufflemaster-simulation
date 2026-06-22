"""Card source abstractions for deterministic and stochastic draws."""

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from random import Random
from typing import Final, Protocol, TypeAlias

from shufflemaster_sim.cards import RANKS, SUITS, Card, Rank, Suit

CardSpec: TypeAlias = Card | tuple[Rank, Suit]
ALLOWED_FINITE_SHOE_DECK_COUNTS: Final[frozenset[int]] = frozenset({1, 4, 6, 8})
MANUAL_SHOE_MIN_ROUND_CARDS: Final[int] = 20


class NoEligibleSlotError(RuntimeError):
    """Raised when no carousel shelf can be ejected."""


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


@dataclass(frozen=True, slots=True)
class One2SixConfig:
    """Configurable assumptions for the One2Six-style source model."""

    deck_count: int = 6
    carousel_slot_count: int = 38
    slot_capacity: int = 10
    output_buffer_target: int = 18
    refill_threshold: int = 8
    min_cards_for_ejection: int = 7
    insertion_rule: str = "uniform_slot_with_capacity"
    output_selection_rule: str = "uniform_eligible_slot"
    ejection_rule: str = "entire_slot"
    input_feed_order: str = "bottom_of_face_up_stack_first"
    accepted_cards_orientation: str = "first_accepted_is_bottom"
    intra_slot_order: str = "lifo"
    ejected_group_order: str = "lifo"
    ingest_policy: str = "instant_on_accept"
    fallback_when_no_eligible_slot: str = "choose_fullest_occupied"
    strict_invariants: bool = True

    def __post_init__(self) -> None:
        if self.deck_count <= 0:
            raise ValueError("deck_count must be positive.")
        if self.carousel_slot_count <= 0:
            raise ValueError("carousel_slot_count must be positive.")
        if self.slot_capacity <= 0:
            raise ValueError("slot_capacity must be positive.")
        if self.output_buffer_target <= 0:
            raise ValueError("output_buffer_target must be positive.")
        if self.refill_threshold < 0:
            raise ValueError("refill_threshold must be non-negative.")
        if self.min_cards_for_ejection <= 0:
            raise ValueError("min_cards_for_ejection must be positive.")
        if self.output_buffer_target <= self.refill_threshold:
            raise ValueError("output_buffer_target must exceed refill_threshold.")
        if self.ejection_rule != "entire_slot":
            raise ValueError("Only entire_slot ejection is implemented.")
        if self.input_feed_order != "bottom_of_face_up_stack_first":
            raise ValueError("Only bottom_of_face_up_stack_first input is implemented.")
        if self.accepted_cards_orientation != "first_accepted_is_bottom":
            raise ValueError(
                "Only first_accepted_is_bottom orientation is implemented."
            )
        if self.intra_slot_order != "lifo":
            raise ValueError("Only lifo intra_slot_order is implemented.")
        if self.ejected_group_order not in {"lifo", "fifo"}:
            raise ValueError("ejected_group_order must be 'lifo' or 'fifo'.")
        if self.ingest_policy != "instant_on_accept":
            raise ValueError("Only instant_on_accept ingest_policy is implemented.")


class One2SixCardSource:
    """Stateful One2Six-style continuous shuffler card source."""

    def __init__(
        self,
        config: One2SixConfig | None = None,
        *,
        seed: int | str | bytes | bytearray | None = None,
        rng: Random | None = None,
        start_draw_id: int = 0,
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")

        self.config = config if config is not None else One2SixConfig()
        self._rng = rng if rng is not None else Random(seed)
        self._next_draw_id = start_draw_id
        self._event_sequence = 0
        self._round_robin_next_slot = 0
        self._telemetry: list[dict[str, object]] = []
        self._ejections: list[dict[str, object]] = []
        self.accepted_discards: list[Card] = []
        self.accepted_discard_batches: list[list[Card]] = []

        self._feeder: deque[Card] = deque()
        self._carousel: list[list[Card]] = [
            [] for _ in range(self.config.carousel_slot_count)
        ]
        self._output_buffer: deque[Card] = deque()
        self._known_physical_ids = self._build_initial_pack()
        self._load_initial_pack()
        self._refill_output_buffer(raise_if_empty=True)
        self._check_invariants_if_strict()

    @property
    def cards_in_feeder(self) -> int:
        return len(self._feeder)

    @property
    def cards_in_carousel(self) -> int:
        return sum(len(shelf) for shelf in self._carousel)

    @property
    def cards_in_output_buffer(self) -> int:
        return len(self._output_buffer)

    @property
    def cards_total_known(self) -> int:
        return len(self._known_physical_ids)

    @property
    def carousel_occupancy(self) -> tuple[int, ...]:
        return tuple(len(shelf) for shelf in self._carousel)

    @property
    def output_buffer_size(self) -> int:
        return len(self._output_buffer)

    @property
    def accepted_discard_batch_count(self) -> int:
        return len(self.accepted_discard_batches)

    @property
    def ejection_count(self) -> int:
        return len(self._ejections)

    @property
    def fallback_ejection_count(self) -> int:
        return sum(1 for record in self._ejections if record["used_fallback"])

    @property
    def draw_count(self) -> int:
        return self._next_draw_id

    @property
    def output_buffer_cards(self) -> tuple[Card, ...]:
        return tuple(self._output_buffer)

    def before_round(self) -> None:
        """No-op round hook for the continuous shuffler model."""

    def draw_card(self) -> Card:
        """Draw from the front of the output buffer."""
        if not self._output_buffer:
            self._refill_output_buffer(raise_if_empty=True)

        physical_card = self._output_buffer.popleft()
        drawn_card = replace(physical_card, draw_id=self._consume_draw_id())
        self._record_event(
            event_type="drawn_from_output_buffer",
            card=drawn_card,
            buffer_size_after=len(self._output_buffer),
        )

        if len(self._output_buffer) <= self.config.refill_threshold:
            self._refill_output_buffer(raise_if_empty=False)

        self._check_invariants_if_strict()
        return drawn_card

    def accept_discards(self, cards: Sequence[Card]) -> None:
        """Accept ordered discards into the feeder and ingest immediately."""
        batch = list(cards)
        self.accepted_discard_batches.append(batch)
        self.accepted_discards.extend(batch)
        for card in batch:
            self._record_event(event_type="accepted_discard", card=card)
            self._feeder.append(card)
            self._record_event(event_type="entered_feeder", card=card)
        self._feed_all_available_cards()
        self._check_invariants_if_strict()

    def assert_invariants(self, external_cards: Sequence[Card] | None = None) -> None:
        """Assert source-local and optional global physical-card invariants."""
        internal_cards = [
            *self._feeder,
            *(card for shelf in self._carousel for card in shelf),
            *self._output_buffer,
        ]
        internal_ids = [card.physical_id for card in internal_cards]
        if len(internal_ids) != len(set(internal_ids)):
            raise AssertionError("Duplicate physical id inside One2Six source.")
        if any(len(shelf) > self.config.slot_capacity for shelf in self._carousel):
            raise AssertionError("Carousel shelf exceeds configured capacity.")
        buffer_ids = [card.physical_id for card in self._output_buffer]
        if len(buffer_ids) != len(set(buffer_ids)):
            raise AssertionError("Output buffer contains duplicate physical ids.")

        if external_cards is None:
            return

        external_ids = [card.physical_id for card in external_cards]
        all_ids = [*internal_ids, *external_ids]
        if len(all_ids) != len(set(all_ids)):
            raise AssertionError("Duplicate physical id across source and externals.")
        if set(all_ids) != self._known_physical_ids:
            raise AssertionError("Known physical card set is incomplete or changed.")

    def telemetry_records(self) -> list[dict[str, object]]:
        """Return telemetry records as dictionaries."""
        return [dict(record) for record in self._telemetry]

    def ejection_records(self) -> list[dict[str, object]]:
        """Return shelf ejection records as dictionaries."""
        return [dict(record) for record in self._ejections]

    def _load_initial_pack(self) -> None:
        initial_pack = self._build_physical_cards(self.config.deck_count)
        self._rng.shuffle(initial_pack)
        self._feeder.extend(initial_pack)
        self._feed_all_available_cards()

    def _feed_all_available_cards(self) -> None:
        while self._feeder:
            card = self._feeder.popleft()
            shelf_id = self._select_insertion_slot()
            self._carousel[shelf_id].append(card)
            self._record_event(
                event_type="entered_carousel_shelf",
                card=card,
                shelf_id=shelf_id,
                shelf_size_after=len(self._carousel[shelf_id]),
            )

    def _select_insertion_slot(self) -> int:
        available_slots = [
            index
            for index, shelf in enumerate(self._carousel)
            if len(shelf) < self.config.slot_capacity
        ]
        if not available_slots:
            raise RuntimeError("No One2Six carousel shelf has available capacity.")
        if self.config.insertion_rule == "uniform_slot_with_capacity":
            return self._rng.choice(available_slots)
        if self.config.insertion_rule == "round_robin_for_testing":
            for _ in range(self.config.carousel_slot_count):
                shelf_id = self._round_robin_next_slot
                self._round_robin_next_slot = (
                    self._round_robin_next_slot + 1
                ) % self.config.carousel_slot_count
                if shelf_id in available_slots:
                    return shelf_id
        if self.config.insertion_rule == "fixed_slot_for_testing":
            if 0 in available_slots:
                return 0
        raise ValueError(f"Unsupported insertion_rule: {self.config.insertion_rule}")

    def _refill_output_buffer(self, *, raise_if_empty: bool) -> None:
        while len(self._output_buffer) < self.config.output_buffer_target:
            selected_slot = self._select_ejection_slot()
            if selected_slot is None:
                break
            self._eject_shelf(selected_slot)

        if raise_if_empty and not self._output_buffer:
            raise NoEligibleSlotError("One2Six output buffer cannot be refilled.")

    def _select_ejection_slot(self) -> int | None:
        eligible_slots = [
            index
            for index, shelf in enumerate(self._carousel)
            if len(shelf) >= self.config.min_cards_for_ejection
        ]
        if eligible_slots:
            if self.config.output_selection_rule == "uniform_eligible_slot":
                return self._rng.choice(eligible_slots)
            if self.config.output_selection_rule == "first_eligible_for_testing":
                return eligible_slots[0]
            if self.config.output_selection_rule == "fixed_slot_for_testing":
                return 0 if 0 in eligible_slots else eligible_slots[0]
            rule = self.config.output_selection_rule
            raise ValueError(f"Unsupported output_selection_rule: {rule}")

        occupied_slots = [
            index for index, shelf in enumerate(self._carousel) if len(shelf) > 0
        ]
        if not occupied_slots:
            return None
        if self.config.fallback_when_no_eligible_slot == "raise":
            raise NoEligibleSlotError("No eligible One2Six shelf is available.")
        if self.config.fallback_when_no_eligible_slot == "choose_fullest_occupied":
            return max(occupied_slots, key=lambda index: len(self._carousel[index]))
        raise ValueError(
            "Unsupported fallback_when_no_eligible_slot: "
            f"{self.config.fallback_when_no_eligible_slot}"
        )

    def _eject_shelf(self, shelf_id: int) -> None:
        shelf = self._carousel[shelf_id]
        used_fallback = len(shelf) < self.config.min_cards_for_ejection
        if self.config.ejected_group_order == "lifo":
            ejected = list(reversed(shelf))
        else:
            ejected = list(shelf)
        shelf.clear()
        buffer_size_before = len(self._output_buffer)
        self._output_buffer.extend(ejected)
        record = {
            "event_sequence": self._next_event_sequence(),
            "shelf_id": shelf_id,
            "group_size": len(ejected),
            "physical_ids": [card.physical_id for card in ejected],
            "used_fallback": used_fallback,
            "buffer_size_before": buffer_size_before,
            "buffer_size_after": len(self._output_buffer),
        }
        self._ejections.append(record)
        for card in ejected:
            self._record_event(
                event_type="entered_output_buffer",
                card=card,
                shelf_id=shelf_id,
                buffer_size_after=len(self._output_buffer),
            )

    def _record_event(
        self,
        *,
        event_type: str,
        card: Card,
        shelf_id: int | None = None,
        shelf_size_after: int | None = None,
        buffer_size_after: int | None = None,
    ) -> None:
        self._telemetry.append(
            {
                "event_sequence": self._next_event_sequence(),
                "event_type": event_type,
                "physical_id": card.physical_id,
                "rank": card.rank,
                "suit": card.suit,
                "draw_id": card.draw_id,
                "shelf_id": shelf_id,
                "shelf_size_after": shelf_size_after,
                "buffer_size_after": buffer_size_after,
            }
        )

    def _check_invariants_if_strict(self) -> None:
        if self.config.strict_invariants:
            self.assert_invariants()

    def _next_event_sequence(self) -> int:
        event_sequence = self._event_sequence
        self._event_sequence += 1
        return event_sequence

    def _consume_draw_id(self) -> int:
        draw_id = self._next_draw_id
        self._next_draw_id += 1
        return draw_id

    def _build_initial_pack(self) -> set[str]:
        return {
            card.physical_id
            for card in self._build_physical_cards(self.config.deck_count)
        }

    def _build_physical_cards(self, deck_count: int) -> list[Card]:
        return [
            Card(
                rank=rank,
                suit=suit,
                physical_id=f"one2six-deck-{deck_index}:{rank}:{suit}",
                draw_id=-1,
            )
            for deck_index in range(deck_count)
            for suit in SUITS
            for rank in RANKS
        ]
