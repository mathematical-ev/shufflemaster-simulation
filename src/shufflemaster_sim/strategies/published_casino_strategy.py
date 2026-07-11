# SPDX-License-Identifier: GPL-3.0-or-later

"""Approximate published S17 multi-deck strategy constrained by house rules."""

from typing import Final

from shufflemaster_sim.actions import ActionType, GameAction
from shufflemaster_sim.cards import Rank
from shufflemaster_sim.hand_values import hand_value_from_ranks, split_value_from_rank
from shufflemaster_sim.state import BlackjackDecisionState

DEALER_UPCARDS: Final[tuple[str, ...]] = (
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "A",
)

HARD_TOTALS: Final[dict[str, tuple[str, ...]]] = {
    "5": ("H", "H", "H", "H", "H", "H", "H", "H", "H", "H"),
    "6": ("H", "H", "H", "H", "H", "H", "H", "H", "H", "H"),
    "7": ("H", "H", "H", "H", "H", "H", "H", "H", "H", "H"),
    "8": ("H", "H", "H", "H", "H", "H", "H", "H", "H", "H"),
    "9": ("H", "D", "D", "D", "D", "H", "H", "H", "H", "H"),
    "10": ("D", "D", "D", "D", "D", "D", "D", "D", "H", "H"),
    "11": ("D", "D", "D", "D", "D", "D", "D", "D", "D", "H"),
    "12": ("H", "H", "S", "S", "S", "H", "H", "H", "H", "H"),
    "13": ("S", "S", "S", "S", "S", "H", "H", "H", "H", "H"),
    "14": ("S", "S", "S", "S", "S", "H", "H", "H", "H", "H"),
    "15": ("S", "S", "S", "S", "S", "H", "H", "H", "H", "H"),
    "16": ("S", "S", "S", "S", "S", "H", "H", "H", "H", "H"),
    "17+": ("S", "S", "S", "S", "S", "S", "S", "S", "S", "S"),
}

SOFT_TOTALS: Final[dict[int, tuple[str, ...]]] = {
    13: ("H", "H", "H", "D", "D", "H", "H", "H", "H", "H"),
    14: ("H", "H", "H", "D", "D", "H", "H", "H", "H", "H"),
    15: ("H", "H", "D", "D", "D", "H", "H", "H", "H", "H"),
    16: ("H", "H", "D", "D", "D", "H", "H", "H", "H", "H"),
    17: ("H", "D", "D", "D", "D", "H", "H", "H", "H", "H"),
    18: ("S", "DS", "DS", "DS", "DS", "S", "S", "H", "H", "H"),
    19: ("S", "S", "S", "S", "S", "S", "S", "S", "S", "S"),
    20: ("S", "S", "S", "S", "S", "S", "S", "S", "S", "S"),
}

PAIR_TOTALS: Final[dict[int | str, tuple[str, ...]]] = {
    2: ("P", "P", "P", "P", "P", "P", "H", "H", "H", "H"),
    3: ("P", "P", "P", "P", "P", "P", "H", "H", "H", "H"),
    4: ("H", "H", "H", "P", "P", "H", "H", "H", "H", "H"),
    5: ("D", "D", "D", "D", "D", "D", "D", "D", "H", "H"),
    6: ("P", "P", "P", "P", "P", "H", "H", "H", "H", "H"),
    7: ("P", "P", "P", "P", "P", "P", "H", "H", "H", "H"),
    8: ("P", "P", "P", "P", "P", "P", "P", "P", "P", "P"),
    9: ("P", "P", "P", "P", "P", "S", "P", "P", "S", "S"),
    10: ("S", "S", "S", "S", "S", "S", "S", "S", "S", "S"),
    "A": ("P", "P", "P", "P", "P", "P", "P", "P", "P", "P"),
}


class PublishedApproxCasinoStrategy:
    """Published S17 multi-deck basic strategy constrained by house rules.

    This is a starting baseline, not an exact solver-generated house-rule-optimal
    strategy.
    """

    def choose_action(
        self,
        *,
        decision: BlackjackDecisionState,
    ) -> GameAction:
        """Choose the best table action that is legal under house rules."""
        dealer_key = self._dealer_key(decision.dealer_upcard_rank)
        strategy_code = self._strategy_code(
            decision.player_ranks,
            dealer_key,
            decision.legal_actions,
        )
        action_type = self._resolve_code(
            strategy_code,
            decision.player_ranks,
            dealer_key,
            decision.legal_actions,
        )
        return GameAction(action_type=action_type)

    def wants_insurance(self) -> bool:
        """Return whether this baseline takes insurance."""
        return False

    def wants_even_money(self) -> bool:
        """Return whether this baseline takes even money."""
        return False

    def _strategy_code(
        self,
        player_ranks: tuple[Rank, ...],
        dealer_key: str,
        legal_actions: frozenset[ActionType],
    ) -> str:
        dealer_index = DEALER_UPCARDS.index(dealer_key)

        if len(player_ranks) == 2 and split_value_from_rank(
            player_ranks[0]
        ) == split_value_from_rank(player_ranks[1]):
            pair_code = PAIR_TOTALS[split_value_from_rank(player_ranks[0])][
                dealer_index
            ]
            if pair_code != "P" or ActionType.SPLIT in legal_actions:
                return pair_code

        return self._total_strategy_code(player_ranks, dealer_key)

    def _total_strategy_code(
        self,
        player_ranks: tuple[Rank, ...],
        dealer_key: str,
    ) -> str:
        dealer_index = DEALER_UPCARDS.index(dealer_key)
        value = hand_value_from_ranks(player_ranks)
        if value.is_soft and 13 <= value.total <= 20:
            return SOFT_TOTALS[value.total][dealer_index]
        if value.total >= 17:
            return HARD_TOTALS["17+"][dealer_index]
        return HARD_TOTALS[str(max(value.total, 5))][dealer_index]

    def _resolve_code(
        self,
        code: str,
        player_ranks: tuple[Rank, ...],
        dealer_key: str,
        legal_actions: frozenset[ActionType],
    ) -> ActionType:
        if code == "P":
            if ActionType.SPLIT in legal_actions:
                return ActionType.SPLIT
            return self._resolve_code(
                self._total_strategy_code(player_ranks, dealer_key),
                player_ranks,
                dealer_key,
                legal_actions,
            )
        if code == "D":
            return self._first_legal(
                (ActionType.DOUBLE, ActionType.HIT, ActionType.STAND),
                legal_actions,
            )
        if code == "DS":
            return self._first_legal(
                (ActionType.DOUBLE, ActionType.STAND, ActionType.HIT),
                legal_actions,
            )
        if code == "S":
            return self._first_legal((ActionType.STAND, ActionType.HIT), legal_actions)
        return self._first_legal((ActionType.HIT, ActionType.STAND), legal_actions)

    def _first_legal(
        self,
        preferred_actions: tuple[ActionType, ...],
        legal_actions: frozenset[ActionType],
    ) -> ActionType:
        for action_type in preferred_actions:
            if action_type in legal_actions:
                return action_type
        raise ValueError("Strategy received no usable legal action.")

    def _dealer_key(self, dealer_upcard_rank: Rank) -> str:
        if dealer_upcard_rank == "A":
            return "A"
        if split_value_from_rank(dealer_upcard_rank) == 10:
            return "T"
        return dealer_upcard_rank
