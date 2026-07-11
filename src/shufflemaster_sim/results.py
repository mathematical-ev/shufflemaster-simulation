# SPDX-License-Identifier: GPL-3.0-or-later

"""Result records and aggregation for simulations."""

from dataclasses import dataclass, field
from typing import Any

from shufflemaster_sim.hand_values import is_natural_blackjack
from shufflemaster_sim.state import BoxState, HandState, TableState


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Result summary for one completed round."""

    round_index: int
    box_count: int
    initial_wagered: float
    action_wagered: float
    total_wagered: float
    net_profit: float
    cumulative_profit_after_round: float


@dataclass(slots=True)
class BoxResult:
    """Aggregate result for one box over many rounds."""

    box_id: int
    rounds_played: int = 0
    hands_resolved: int = 0
    blackjacks: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    busts: int = 0
    doubles: int = 0
    splits: int = 0
    initial_wagered: float = 0.0
    action_wagered: float = 0.0
    total_wagered: float = 0.0
    net_profit: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0


@dataclass(slots=True)
class SimulationResult:
    """Structured result for a completed simulation."""

    rounds_played: int
    round_results: list[RoundResult]
    box_results: list[BoxResult]
    total_hands_resolved: int
    initial_wagered: float
    action_wagered: float
    total_wagered: float
    net_profit: float
    total_net_profit: float
    cumulative_profit: float
    average_profit_per_round: float
    average_profit_per_initial_wager: float
    edge_per_initial_wager: float
    edge_per_total_wager: float
    final_bankroll_delta: float
    shuffle_count: int | None = None
    output_buffer_size: int | None = None
    ejection_count: int | None = None
    fallback_ejection_count: int | None = None

    def as_round_records(self) -> list[dict[str, Any]]:
        """Return round results as plain records for later analysis."""
        return [
            {
                "round_index": round_result.round_index,
                "box_count": round_result.box_count,
                "initial_wagered": round_result.initial_wagered,
                "action_wagered": round_result.action_wagered,
                "total_wagered": round_result.total_wagered,
                "net_profit": round_result.net_profit,
                "cumulative_profit_after_round": (
                    round_result.cumulative_profit_after_round
                ),
            }
            for round_result in self.round_results
        ]

    def as_box_records(self) -> list[dict[str, Any]]:
        """Return box results as plain records for later analysis."""
        return [
            {
                "box_id": box.box_id,
                "rounds_played": box.rounds_played,
                "hands_resolved": box.hands_resolved,
                "blackjacks": box.blackjacks,
                "wins": box.wins,
                "losses": box.losses,
                "pushes": box.pushes,
                "busts": box.busts,
                "doubles": box.doubles,
                "splits": box.splits,
                "initial_wagered": box.initial_wagered,
                "action_wagered": box.action_wagered,
                "total_wagered": box.total_wagered,
                "net_profit": box.net_profit,
                "current_win_streak": box.current_win_streak,
                "current_loss_streak": box.current_loss_streak,
                "max_win_streak": box.max_win_streak,
                "max_loss_streak": box.max_loss_streak,
            }
            for box in self.box_results
        ]


@dataclass(slots=True)
class ResultRecorder:
    """Accumulates table states into simulation results."""

    base_bet: float
    box_count: int
    round_results: list[RoundResult] = field(default_factory=list)
    box_results: dict[int, BoxResult] = field(default_factory=dict)
    cumulative_profit: float = 0.0
    retain_round_results: bool = True
    rounds_recorded: int = 0

    def record_round(self, table: TableState) -> RoundResult:
        """Record one settled table round."""
        round_initial_wagered = sum(box.base_bet for box in table.boxes)
        round_total_wagered = sum(
            self._hand_amount_wagered(hand) for box in table.boxes for hand in box.hands
        )
        round_action_wagered = round_total_wagered - round_initial_wagered
        round_profit = sum(hand.net_result for box in table.boxes for hand in box.hands)
        self.cumulative_profit += round_profit
        round_result = RoundResult(
            round_index=table.round_index,
            box_count=len(table.boxes),
            initial_wagered=round_initial_wagered,
            action_wagered=round_action_wagered,
            total_wagered=round_total_wagered,
            net_profit=round_profit,
            cumulative_profit_after_round=self.cumulative_profit,
        )
        self.rounds_recorded += 1
        if self.retain_round_results:
            self.round_results.append(round_result)

        for box in table.boxes:
            self._record_box(box)

        return round_result

    def build_result(self) -> SimulationResult:
        """Build an immutable summary for the completed simulation."""
        total_hands = sum(box.hands_resolved for box in self.box_results.values())
        initial_wagered = sum(box.initial_wagered for box in self.box_results.values())
        action_wagered = sum(box.action_wagered for box in self.box_results.values())
        total_wagered = sum(box.total_wagered for box in self.box_results.values())
        rounds_played = self.rounds_recorded
        total_net_profit = self.cumulative_profit
        edge_per_initial_wager = (
            total_net_profit / initial_wagered if initial_wagered else 0.0
        )
        edge_per_total_wager = (
            total_net_profit / total_wagered if total_wagered else 0.0
        )
        return SimulationResult(
            rounds_played=rounds_played,
            round_results=list(self.round_results),
            box_results=[self.box_results[key] for key in sorted(self.box_results)],
            total_hands_resolved=total_hands,
            initial_wagered=initial_wagered,
            action_wagered=action_wagered,
            total_wagered=total_wagered,
            net_profit=total_net_profit,
            total_net_profit=total_net_profit,
            cumulative_profit=self.cumulative_profit,
            average_profit_per_round=(
                total_net_profit / rounds_played if rounds_played else 0.0
            ),
            average_profit_per_initial_wager=edge_per_initial_wager,
            edge_per_initial_wager=edge_per_initial_wager,
            edge_per_total_wager=edge_per_total_wager,
            final_bankroll_delta=total_net_profit,
        )

    def _record_box(self, box: BoxState) -> None:
        result = self.box_results.setdefault(box.box_id, BoxResult(box_id=box.box_id))
        round_net = sum(hand.net_result for hand in box.hands)
        result.rounds_played += 1
        result.hands_resolved += len(box.hands)
        result.splits += 1 if len(box.hands) > 1 else 0
        result.initial_wagered += box.base_bet

        for hand in box.hands:
            result.blackjacks += 1 if self._is_blackjack(hand) else 0
            result.wins += 1 if self._is_win(hand) else 0
            result.losses += 1 if self._is_loss(hand) else 0
            result.pushes += 1 if self._is_push(hand) else 0
            result.busts += 1 if hand.outcome_label == "bust" else 0
            result.doubles += 1 if hand.is_doubled else 0
            result.total_wagered += self._hand_amount_wagered(hand)
            result.net_profit += hand.net_result

        result.action_wagered = result.total_wagered - result.initial_wagered
        self._record_streak(result, round_net)

    def _hand_amount_wagered(self, hand: HandState) -> float:
        return hand.wager * (2.0 if hand.is_doubled else 1.0)

    def _is_win(self, hand: HandState) -> bool:
        return hand.net_result > 0

    def _is_loss(self, hand: HandState) -> bool:
        return hand.net_result < 0

    def _is_push(self, hand: HandState) -> bool:
        return hand.outcome_label is not None and hand.net_result == 0.0

    def _is_blackjack(self, hand: HandState) -> bool:
        return is_natural_blackjack(
            hand.cards,
            blackjack_eligible=hand.blackjack_eligible,
        )

    def _record_streak(self, result: BoxResult, round_net: float) -> None:
        if round_net > 0:
            result.current_win_streak += 1
            result.current_loss_streak = 0
        elif round_net < 0:
            result.current_loss_streak += 1
            result.current_win_streak = 0

        result.max_win_streak = max(
            result.max_win_streak,
            result.current_win_streak,
        )
        result.max_loss_streak = max(
            result.max_loss_streak,
            result.current_loss_streak,
        )
