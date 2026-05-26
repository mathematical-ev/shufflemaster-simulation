from shufflemaster_sim.cards import Card
from shufflemaster_sim.results import ResultRecorder
from shufflemaster_sim.state import BoxState, DealerState, HandState, TableState


def settled_table(round_index: int, net_result: float) -> TableState:
    return TableState(
        boxes=[
            BoxState(
                box_id=1,
                base_bet=10.0,
                hands=[
                    HandState(
                        hand_id=0,
                        cards=[],
                        wager=10.0,
                        is_terminal=True,
                        outcome_label="win" if net_result > 0 else "loss",
                        net_result=net_result,
                    )
                ],
            )
        ],
        dealer=DealerState(),
        round_index=round_index,
    )


def settled_box(box_id: int, net_result: float) -> BoxState:
    return BoxState(
        box_id=box_id,
        base_bet=10.0,
        hands=[
            HandState(
                hand_id=0,
                cards=[],
                wager=10.0,
                is_terminal=True,
                outcome_label="win" if net_result > 0 else "loss",
                net_result=net_result,
            )
        ],
    )


def settled_multi_box_table(
    round_index: int,
    net_results: dict[int, float],
) -> TableState:
    return TableState(
        boxes=[
            settled_box(box_id, net_result)
            for box_id, net_result in sorted(net_results.items())
        ],
        dealer=DealerState(),
        round_index=round_index,
    )


def test_result_recorder_cumulative_profit_equals_sum_of_round_profits() -> None:
    recorder = ResultRecorder(base_bet=10.0, box_count=1)
    round_results = [
        recorder.record_round(settled_table(0, 10.0)),
        recorder.record_round(settled_table(1, -20.0)),
        recorder.record_round(settled_table(2, 0.0)),
    ]
    result = recorder.build_result()

    assert result.cumulative_profit == sum(round.net_profit for round in round_results)
    assert result.total_net_profit == -10.0
    assert result.net_profit == -10.0
    assert result.final_bankroll_delta == -10.0


def test_result_recorder_counts_blackjack_pushes_as_blackjacks() -> None:
    table = TableState(
        boxes=[
            BoxState(
                box_id=1,
                base_bet=10.0,
                hands=[
                    HandState(
                        hand_id=0,
                        cards=[
                            Card(
                                rank="A",
                                suit="spades",
                                physical_id="test:0",
                                draw_id=0,
                            ),
                            Card(
                                rank="T",
                                suit="spades",
                                physical_id="test:1",
                                draw_id=1,
                            ),
                        ],
                        wager=10.0,
                        is_terminal=True,
                        outcome_label="blackjack_push",
                        net_result=0.0,
                    )
                ],
            )
        ],
        dealer=DealerState(),
        round_index=0,
    )
    recorder = ResultRecorder(base_bet=10.0, box_count=1)

    recorder.record_round(table)
    result = recorder.build_result()

    assert result.box_results[0].blackjacks == 1
    assert result.box_results[0].pushes == 1


def test_result_recorder_separates_initial_action_and_total_wagers() -> None:
    table = TableState(
        boxes=[
            BoxState(
                box_id=1,
                base_bet=10.0,
                hands=[
                    HandState(
                        hand_id=0,
                        cards=[],
                        wager=10.0,
                        is_doubled=True,
                        outcome_label="win",
                        net_result=20.0,
                    ),
                    HandState(
                        hand_id=1,
                        cards=[],
                        wager=10.0,
                        is_split_hand=True,
                        outcome_label="push",
                        net_result=0.0,
                    ),
                ],
            )
        ],
        dealer=DealerState(),
        round_index=0,
    )
    recorder = ResultRecorder(base_bet=10.0, box_count=1)

    round_result = recorder.record_round(table)
    result = recorder.build_result()

    assert round_result.initial_wagered == 10.0
    assert round_result.action_wagered == 20.0
    assert round_result.total_wagered == 30.0
    assert result.initial_wagered == 10.0
    assert result.action_wagered == 20.0
    assert result.total_wagered == 30.0
    assert result.edge_per_initial_wager == 2.0
    assert result.edge_per_total_wager == 20.0 / 30.0


def test_win_win_loss_tracks_win_and_current_loss_streaks() -> None:
    recorder = ResultRecorder(base_bet=10.0, box_count=1)
    recorder.record_round(settled_table(0, 10.0))
    recorder.record_round(settled_table(1, 10.0))
    recorder.record_round(settled_table(2, -10.0))

    box = recorder.build_result().box_results[0]

    assert box.max_win_streak == 2
    assert box.current_loss_streak == 1


def test_loss_loss_push_loss_tracks_loss_streak_reset() -> None:
    recorder = ResultRecorder(base_bet=10.0, box_count=1)
    recorder.record_round(settled_table(0, -10.0))
    recorder.record_round(settled_table(1, -10.0))
    recorder.record_round(settled_table(2, 0.0))
    recorder.record_round(settled_table(3, -10.0))

    box = recorder.build_result().box_results[0]

    assert box.max_loss_streak == 2
    assert box.current_loss_streak == 1
    assert box.current_win_streak == 0


def test_push_resets_both_current_streaks() -> None:
    recorder = ResultRecorder(base_bet=10.0, box_count=1)
    recorder.record_round(settled_table(0, 10.0))
    recorder.record_round(settled_table(1, 0.0))

    box = recorder.build_result().box_results[0]

    assert box.current_win_streak == 0
    assert box.current_loss_streak == 0


def test_streaks_are_independent_by_box_id() -> None:
    recorder = ResultRecorder(base_bet=10.0, box_count=2)
    recorder.record_round(settled_multi_box_table(0, {1: 10.0, 2: -10.0}))
    recorder.record_round(settled_multi_box_table(1, {1: 10.0, 2: -10.0}))
    recorder.record_round(settled_multi_box_table(2, {1: -10.0, 2: 10.0}))

    box_1, box_2 = recorder.build_result().box_results

    assert box_1.max_win_streak == 2
    assert box_1.current_loss_streak == 1
    assert box_2.max_loss_streak == 2
    assert box_2.current_win_streak == 1


def test_total_result_aggregation_does_not_corrupt_per_box_streaks() -> None:
    recorder = ResultRecorder(base_bet=10.0, box_count=2)
    recorder.record_round(settled_multi_box_table(0, {1: 10.0, 2: -10.0}))
    recorder.record_round(settled_multi_box_table(1, {1: 10.0, 2: -10.0}))

    result = recorder.build_result()

    assert result.net_profit == 0.0
    assert result.box_results[0].max_win_streak == 2
    assert result.box_results[1].max_loss_streak == 2
