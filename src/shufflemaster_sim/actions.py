# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko

"""Player action types for blackjack-like games."""

from dataclasses import dataclass
from enum import StrEnum


class ActionType(StrEnum):
    """Supported player decisions."""

    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    INSURANCE = "insurance"
    EVEN_MONEY = "even_money"


@dataclass(frozen=True, slots=True)
class GameAction:
    """A player action scoped to an optional box and hand."""

    action_type: ActionType
    box_id: int | None = None
    hand_id: int | None = None
