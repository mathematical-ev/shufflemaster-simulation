# SPDX-License-Identifier: GPL-3.0-or-later

"""Casino Blackjack rules and one-round engine."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from shufflemaster_sim.actions import ActionType, GameAction
from shufflemaster_sim.card_sources import CardSource
from shufflemaster_sim.cards import Card
from shufflemaster_sim.hand_values import (
    hand_value,
    is_bust,
    is_double_eligible_total,
    is_natural_blackjack,
    split_value,
)
from shufflemaster_sim.state import (
    BlackjackDecisionState,
    BoxState,
    DealerState,
    HandState,
    TableState,
)


class CasinoBlackjackStrategy(Protocol):
    """Protocol for strategies that choose from legal house-rule actions."""

    def choose_action(
        self,
        *,
        decision: BlackjackDecisionState,
    ) -> GameAction:
        """Choose the next action."""


@dataclass(frozen=True, slots=True)
class CasinoBlackjackConfig:
    """Configuration for Casino Blackjack metadata and table rules."""

    base_bet: float = 10.0
    box_count: int = 1
    box_bets: Mapping[int, float] | None = None
    deck_count: int = 6
    blackjack_payout: float = 1.5
    dealer_hits_soft_17: bool = False
    allow_resplit: bool = False
    max_hands_per_box: int = 2
    use_shuffling_device: bool = True
    burn_initial_card: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.box_count <= 7:
            raise ValueError("box_count must be between 1 and 7.")
        if self.base_bet <= 0:
            raise ValueError("base_bet must be positive.")
        if any(bet <= 0 for bet in self.resolved_box_bets().values()):
            raise ValueError("all box bets must be positive.")
        if self.deck_count not in {1, 4, 6, 8}:
            raise ValueError("deck_count metadata must be one of: 1, 4, 6, 8.")
        if self.blackjack_payout <= 0:
            raise ValueError("blackjack_payout must be positive.")
        if self.max_hands_per_box < 2:
            raise ValueError("max_hands_per_box must be at least 2.")

    def resolved_box_bets(self) -> dict[int, float]:
        """Return independent base wagers for configured boxes."""
        if self.box_bets is None:
            return {box_id: self.base_bet for box_id in range(1, self.box_count + 1)}
        box_bets = dict(self.box_bets)
        expected_box_ids = set(range(1, self.box_count + 1))
        if set(box_bets) != expected_box_ids:
            raise ValueError("box_bets must provide exactly one bet per active box.")
        return box_bets


class CasinoBlackjackGame:
    """Rules engine for one round at a Casino Blackjack table."""

    def __init__(self, config: CasinoBlackjackConfig | None = None) -> None:
        self.config = config if config is not None else CasinoBlackjackConfig()
        self._pending_discard_rack: list[Card] = []
        self._initial_burn_completed = False

    @property
    def pending_discard_rack(self) -> tuple[Card, ...]:
        """Return cards awaiting the next post-initial-deal source return."""
        return tuple(self._pending_discard_rack)

    def play_round(
        self,
        *,
        round_index: int,
        card_source: CardSource,
        strategy: CasinoBlackjackStrategy,
    ) -> TableState:
        """Play and settle one full round."""
        card_source.before_round()
        self.burn_initial_card(card_source)
        table = self.create_table(round_index)
        self.deal_initial_cards(table, card_source)
        self.settle_immediate_blackjacks(table)
        self.play_player_hands(table, card_source, strategy)
        self.play_dealer(table, card_source)
        self.settle(table)
        self.collect_remaining_layout_cards(table)
        self.stage_discard_rack_for_next_round(table)
        return table

    def burn_initial_card(self, card_source: CardSource) -> None:
        """Burn the first source card once at the start of a game session."""
        if not self.config.burn_initial_card or self._initial_burn_completed:
            return
        self._pending_discard_rack.append(card_source.draw_card())
        self._initial_burn_completed = True

    def create_table(self, round_index: int) -> TableState:
        """Create empty round state with configured boxes."""
        box_bets = self.config.resolved_box_bets()
        boxes = [
            BoxState(
                box_id=box_id,
                base_bet=box_bets[box_id],
                hands=[
                    HandState(
                        hand_id=0,
                        cards=[],
                        wager=box_bets[box_id],
                    )
                ],
            )
            for box_id in range(1, self.config.box_count + 1)
        ]
        return TableState(
            boxes=boxes,
            dealer=DealerState(),
            round_index=round_index,
        )

    def deal_initial_cards(self, table: TableState, card_source: CardSource) -> None:
        """Deal first player cards, dealer upcard, then second player cards."""
        for box in table.boxes:
            box.hands[0].cards.append(card_source.draw_card())

        table.dealer.cards.append(card_source.draw_card())

        for box in table.boxes:
            box.hands[0].cards.append(card_source.draw_card())
            if is_natural_blackjack(box.hands[0].cards):
                box.hands[0].is_terminal = True

        self.return_pending_discards_after_initial_deal(card_source)

    def return_pending_discards_after_initial_deal(
        self,
        card_source: CardSource,
    ) -> None:
        """Return the previous round rack after the next initial deal."""
        if not self.config.use_shuffling_device or not self._pending_discard_rack:
            return
        card_source.accept_discards(list(self._pending_discard_rack))
        self._pending_discard_rack.clear()

    def settle_immediate_blackjacks(self, table: TableState) -> None:
        """Pay and collect player blackjacks when dealer cannot have blackjack."""
        dealer_upcard = table.dealer.cards[0]
        if split_value(dealer_upcard) in {"A", 10}:
            return

        for box in table.boxes:
            for hand in box.hands:
                if is_natural_blackjack(
                    hand.cards,
                    blackjack_eligible=hand.blackjack_eligible,
                ):
                    self._settle_hand(
                        hand,
                        label="blackjack_win",
                        net_result=self.config.blackjack_payout * hand.wager,
                    )
                    self.collect_hand_cards(table, hand)

    def legal_actions(
        self,
        *,
        table: TableState,
        box: BoxState,
        hand: HandState,
    ) -> frozenset[ActionType]:
        """Return legal actions for a hand under Casino Blackjack constraints."""
        _ = table
        if hand.outcome_label is not None or hand.is_from_split_aces:
            return frozenset()
        if is_bust(hand.cards):
            return frozenset()

        value = hand_value(hand.cards)
        if value.total >= 21:
            return frozenset({ActionType.STAND})

        actions: set[ActionType] = {ActionType.HIT}

        if value.total >= 12:
            actions.add(ActionType.STAND)

        if self._can_double(hand):
            actions.add(ActionType.DOUBLE)

        if self._can_split(box, hand):
            actions.add(ActionType.SPLIT)

        return frozenset(actions)

    def play_player_hands(
        self,
        table: TableState,
        card_source: CardSource,
        strategy: CasinoBlackjackStrategy,
    ) -> None:
        """Play all boxes and hands left-to-right."""
        dealer_upcard = table.dealer.cards[0]
        for box in table.boxes:
            hand_index = 0
            while hand_index < len(box.hands):
                hand = box.hands[hand_index]
                while self._hand_needs_player_action(hand):
                    legal_actions = self.legal_actions(
                        table=table,
                        box=box,
                        hand=hand,
                    )
                    if not legal_actions:
                        hand.is_terminal = True
                        break

                    action = strategy.choose_action(
                        decision=BlackjackDecisionState(
                            player_ranks=tuple(card.rank for card in hand.cards),
                            dealer_upcard_rank=dealer_upcard.rank,
                            legal_actions=legal_actions,
                            is_split_hand=hand.is_split_hand,
                        )
                    )
                    self.apply_action(
                        table=table,
                        box=box,
                        hand=hand,
                        action_type=action.action_type,
                        card_source=card_source,
                    )

                    if action.action_type == ActionType.SPLIT:
                        hand = box.hands[hand_index]

                hand_index += 1

    def apply_action(
        self,
        *,
        table: TableState,
        box: BoxState,
        hand: HandState,
        action_type: ActionType,
        card_source: CardSource,
    ) -> None:
        """Apply one legal player action."""
        legal_actions = self.legal_actions(table=table, box=box, hand=hand)
        if action_type not in legal_actions:
            raise ValueError(f"Illegal action {action_type.value!r}.")

        if action_type == ActionType.HIT:
            self._hit(table, hand, card_source)
            return

        if action_type == ActionType.STAND:
            hand.is_terminal = True
            return

        if action_type == ActionType.DOUBLE:
            hand.is_doubled = True
            self._hit(table, hand, card_source)
            if hand.outcome_label is None:
                hand.is_terminal = True
            return

        if action_type == ActionType.SPLIT:
            self._split_hand(box, hand, card_source)
            return

        raise ValueError(f"Unsupported action {action_type.value!r}.")

    def play_dealer(
        self,
        table: TableState,
        card_source: CardSource,
    ) -> None:
        """Play the dealer if at least one hand still needs comparison."""
        if not self._needs_dealer_resolution(table):
            return

        if self._only_unresolved_player_blackjacks(table):
            if len(table.dealer.cards) == 1 and self._dealer_can_have_blackjack(table):
                table.dealer.cards.append(card_source.draw_card())
            return

        if len(table.dealer.cards) == 1:
            table.dealer.cards.append(card_source.draw_card())
            if is_natural_blackjack(table.dealer.cards):
                return

        while self._dealer_should_hit(table.dealer.cards):
            table.dealer.cards.append(card_source.draw_card())

    def settle(self, table: TableState) -> None:
        """Settle all unresolved player hands."""
        dealer_blackjack = is_natural_blackjack(table.dealer.cards)
        dealer_bust = is_bust(table.dealer.cards)
        dealer_total = hand_value(table.dealer.cards).total

        for box in table.boxes:
            if dealer_blackjack and self._box_has_unsettled_split_hands(box):
                self._settle_split_box_against_dealer_blackjack(box)
                continue

            for hand in box.hands:
                if hand.outcome_label is not None:
                    continue
                if dealer_blackjack:
                    self._settle_hand_against_dealer_blackjack(hand)
                    continue
                if is_natural_blackjack(
                    hand.cards,
                    blackjack_eligible=hand.blackjack_eligible,
                ):
                    self._settle_hand(
                        hand,
                        label="blackjack_win",
                        net_result=self.config.blackjack_payout * hand.wager,
                    )
                    continue
                if dealer_bust:
                    self._settle_hand(
                        hand,
                        label="win",
                        net_result=self._ordinary_win_amount(hand),
                    )
                    continue

                player_total = hand_value(hand.cards).total
                if player_total > dealer_total:
                    self._settle_hand(
                        hand,
                        label="win",
                        net_result=self._ordinary_win_amount(hand),
                    )
                elif player_total < dealer_total:
                    self._settle_hand(
                        hand,
                        label="loss",
                        net_result=-self._ordinary_win_amount(hand),
                    )
                else:
                    self._settle_hand(hand, label="push", net_result=0.0)

    def collect_hand_cards(self, table: TableState, hand: HandState) -> None:
        """Append one hand's cards to the ordered discard rack once."""
        if hand.is_collected:
            return
        table.discard_rack.extend(hand.cards)
        hand.is_collected = True

    def collect_remaining_layout_cards(self, table: TableState) -> None:
        """Collect uncollected cards in box/hand order, then dealer last."""
        for box in sorted(table.boxes, key=lambda current_box: current_box.box_id):
            for hand in sorted(
                box.hands, key=lambda current_hand: current_hand.hand_id
            ):
                self.collect_hand_cards(table, hand)

        if not table.dealer.is_collected:
            table.discard_rack.extend(table.dealer.cards)
            table.dealer.is_collected = True

        self._validate_no_duplicate_discard_events(table.discard_rack)

    def stage_discard_rack_for_next_round(self, table: TableState) -> None:
        """Make this round's ordered rack the next pending shuffling-device return."""
        self._pending_discard_rack = list(table.discard_rack)

    def _hit(
        self,
        table: TableState,
        hand: HandState,
        card_source: CardSource,
    ) -> None:
        hand.cards.append(card_source.draw_card())
        if is_bust(hand.cards):
            self._settle_hand(
                hand,
                label="bust",
                net_result=-self._ordinary_win_amount(hand),
            )
            self.collect_hand_cards(table, hand)
        elif hand_value(hand.cards).total >= 21:
            hand.is_terminal = True

    def _split_hand(
        self,
        box: BoxState,
        hand: HandState,
        card_source: CardSource,
    ) -> None:
        first_card, second_card = hand.cards
        hand_position = box.hands.index(hand)
        first_hand = HandState(
            hand_id=hand.hand_id,
            cards=[first_card, card_source.draw_card()],
            wager=hand.wager,
            is_split_hand=True,
            blackjack_eligible=False,
            is_from_split_aces=first_card.rank == "A",
        )
        second_hand = HandState(
            hand_id=self._next_hand_id(box),
            cards=[second_card, card_source.draw_card()],
            wager=hand.wager,
            is_split_hand=True,
            blackjack_eligible=False,
            is_from_split_aces=second_card.rank == "A",
        )

        for split_hand in (first_hand, second_hand):
            if (
                split_hand.is_from_split_aces
                or hand_value(split_hand.cards).total >= 21
            ):
                split_hand.is_terminal = True

        box.hands[hand_position : hand_position + 1] = [first_hand, second_hand]

    def _can_double(self, hand: HandState) -> bool:
        return (
            not hand.is_terminal
            and hand.outcome_label is None
            and len(hand.cards) == 2
            and is_double_eligible_total(hand.cards)
        )

    def _can_split(self, box: BoxState, hand: HandState) -> bool:
        if hand.is_terminal or hand.outcome_label is not None:
            return False
        if len(hand.cards) != 2 or len(box.hands) >= self.config.max_hands_per_box:
            return False
        if hand.is_split_hand and not self.config.allow_resplit:
            return False
        return split_value(hand.cards[0]) == split_value(hand.cards[1])

    def _hand_needs_player_action(self, hand: HandState) -> bool:
        if hand.outcome_label is not None or hand.is_terminal:
            return False
        if is_bust(hand.cards):
            return False
        return hand_value(hand.cards).total < 21

    def _dealer_should_hit(self, cards: Sequence[Card]) -> bool:
        if is_bust(list(cards)):
            return False
        value = hand_value(list(cards))
        if value.total < 17:
            return True
        return self.config.dealer_hits_soft_17 and value.total == 17 and value.is_soft

    def _needs_dealer_resolution(self, table: TableState) -> bool:
        return any(
            hand.outcome_label is None for box in table.boxes for hand in box.hands
        )

    def _only_unresolved_player_blackjacks(self, table: TableState) -> bool:
        unresolved_hands = [
            hand
            for box in table.boxes
            for hand in box.hands
            if hand.outcome_label is None
        ]
        return bool(unresolved_hands) and all(
            is_natural_blackjack(
                hand.cards,
                blackjack_eligible=hand.blackjack_eligible,
            )
            for hand in unresolved_hands
        )

    def _dealer_can_have_blackjack(self, table: TableState) -> bool:
        return bool(table.dealer.cards) and split_value(table.dealer.cards[0]) in {
            "A",
            10,
        }

    def _ordinary_win_amount(self, hand: HandState) -> float:
        return hand.wager * (2.0 if hand.is_doubled else 1.0)

    def _settle_hand(
        self,
        hand: HandState,
        *,
        label: str,
        net_result: float,
    ) -> None:
        hand.outcome_label = label
        hand.net_result = net_result
        hand.is_terminal = True

    def _settle_hand_against_dealer_blackjack(self, hand: HandState) -> None:
        if is_natural_blackjack(hand.cards, blackjack_eligible=hand.blackjack_eligible):
            self._settle_hand(hand, label="blackjack_push", net_result=0.0)
        elif hand.is_doubled:
            self._settle_hand(
                hand,
                label="dealer_blackjack_double_loss",
                net_result=-hand.wager,
            )
        else:
            self._settle_hand(
                hand,
                label="dealer_blackjack_loss",
                net_result=-hand.wager,
            )

    def _box_has_unsettled_split_hands(self, box: BoxState) -> bool:
        return any(
            hand.is_split_hand and hand.outcome_label is None for hand in box.hands
        )

    def _settle_split_box_against_dealer_blackjack(self, box: BoxState) -> None:
        original_wager_collected = any(
            hand.is_split_hand and hand.net_result < 0 for hand in box.hands
        )
        for hand in box.hands:
            if hand.outcome_label is not None:
                continue
            if not original_wager_collected:
                self._settle_hand(
                    hand,
                    label="dealer_blackjack_split_original_loss",
                    net_result=-box.base_bet,
                )
                original_wager_collected = True
            else:
                self._settle_hand(
                    hand,
                    label="dealer_blackjack_split_standoff",
                    net_result=0.0,
                )

    def _next_hand_id(self, box: BoxState) -> int:
        return max(hand.hand_id for hand in box.hands) + 1

    def _validate_no_duplicate_discard_events(
        self,
        discard_rack: Sequence[Card],
    ) -> None:
        draw_ids = [card.draw_id for card in discard_rack]
        if len(draw_ids) != len(set(draw_ids)):
            raise RuntimeError("Discard rack contains duplicate draw events.")
