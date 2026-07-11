# shufflemaster-sim

`shufflemaster-sim` is a Python simulation project for blackjack-like card
games. Its long-term purpose is to test whether continuous shuffling machines,
especially Shuffle Master / One2Six-style devices, can create exploitable
short-horizon structure.

The first milestone is deliberately small: a clean project layout and a
foundational card-source layer that can be tested independently from future game
rules.

## Architecture Principles

- Card dealing/card sources are separate from game mechanics.
- Game rules are separate from player actions and strategies.
- Player strategy is separate from legal-action validation.
- Settlement/accounting is separate from card generation.
- Randomness is injectable and reproducible.
- Multiple games and card sources should fit behind explicit interfaces.

## Setup

Activate the existing local virtual environment:

```bash
source .venv/bin/activate
```

Install development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Development Commands

Run tests:

```bash
python -m pytest
```

Run linting:

```bash
python -m ruff check .
```

Check formatting:

```bash
python -m ruff format --check .
```

Apply formatting:

```bash
python -m ruff format .
```

Run type checks:

```bash
python -m mypy src
```

## Casino Blackjack Baseline

Run the one-box IID Casino Blackjack baseline simulation:

```bash
python scripts/run_casino_blackjack_baseline.py --rounds 10000 --base-bet 10 --seed 42 --card-source iid
```

Run the same baseline with a finite shuffled shoe:

```bash
python scripts/run_casino_blackjack_baseline.py --rounds 10000 --base-bet 10 --seed 42 --card-source finite-shoe --deck-count 6
```

Run the manual shoe benchmark:

```bash
python scripts/run_casino_blackjack_baseline.py --rounds 10000 --base-bet 10 --seed 42 --card-source manual-shoe --deck-count 8 --cut-card-penetration 0.75
```

Run the baseline with the configurable One2Six-style source model:

```bash
python scripts/run_casino_blackjack_baseline.py --rounds 10000 --base-bet 10 --seed 42 --card-source one2six
```

Run the One2Six source diagnostic without the game engine:

```bash
python scripts/run_one2six_source_diagnostics.py --draws 100000 --seed 42
```

The baseline uses a configurable base bet and defaults to `IidRandomCardSource`.
It is structured so the card source can be replaced by a finite shoe, manual
shoe, or One2Six-style source without changing the Casino Blackjack game rules.

`PublishedApproxCasinoStrategy` is a starting published S17 multi-deck basic
strategy constrained by Casino Blackjack legal actions. It is not yet a
solver-generated exact Casino Blackjack strategy.

The current casino rule profile stands on every hard or soft 17. It also burns
one card at the start of a game session and returns that burn through the same
post-initial-deal discard path used for later rounds.

The engine explicitly models ordered discard-rack collection and
shuffling-device return timing. Current-round discards are staged for return
only after the next round's initial deal, which matters for future One2Six-style
card-source simulation.

The current runnable simulation supports IID random cards, a generic
finite-shoe source, a manual shoe benchmark, and a configurable One2Six-style
source. It does not make exploitability claims; those require validated source
models and simulation evidence.

The current CLI still runs one box, but internally boxes are represented with
independent wagers such as `box_bets = {1: 10.0}`. Per-box win/loss streaks are
tracked by each box's net result per round. A neutral/push round does not break
the active win or loss streak.

## Card Sources And Identity

`IidRandomCardSource` treats every draw as a new physical card. It is useful for
baseline game-mechanics validation, but it is not a shoe or shuffler model.

`FiniteShoeCardSource` builds a finite set of physical cards, shuffles them, and
draws without replacement. Accepted discards are stored in order and are not
drawable again until the shoe is empty and the explicit reshuffle policy returns
the discard tray to the shoe.

`ManualShoeCardSource` is the manual card-shoe benchmark. It defaults to eight
decks and a configurable 75% cut-card penetration assumption. The cut-card
threshold marks a shuffle for the next round; the source does not reshuffle in
the middle of a round. On shuffle, remaining shoe cards and accepted discards
are combined while preserving physical card identities.

`One2SixCardSource` is a configurable approximation of a continuous shuffling
machine source for return-time experiments. It preserves physical card identity,
assigns a fresh `draw_id` on every draw, accepts ordered discard batches, feeds
accepted cards into a carousel, ejects whole shelves into an output buffer, and
records telemetry for source-level diagnostics. The default settings are
modelling assumptions, not claims about proprietary device internals.

Cards carry two identities:

- `physical_id`: stable identity for the real card moving through shoe, hand,
  discard rack, shuffler, and possible later redraw.
- `draw_id`: unique identity for the specific draw event.

## Experiments

Experiment code lives outside the core package in `experiments/`. The core
engine remains responsible for cards, card sources, game rules, strategies, and
simulation mechanics. Experiments are responsible for controlled runs,
aggregate metrics, baseline comparisons, plots, and output files.

Install optional plotting dependencies with:

```bash
python -m pip install -e ".[dev,analysis]"
```

Run a smoke-sized IID baseline experiment:

```bash
python scripts/run_iid_baseline_experiment.py \
  --source-draws 10000 \
  --game-rounds 1000 \
  --base-bet 10 \
  --seed 42 \
  --output-dir experiments/outputs/iid_smoke
```

Run a larger IID baseline experiment:

```bash
python scripts/run_iid_baseline_experiment.py \
  --source-draws 1000000 \
  --game-rounds 1000000 \
  --base-bet 10 \
  --seed 42 \
  --output-dir experiments/outputs/iid_1m_seed42
```

The IID source experiment tracks rank, suit, rank/suit, Hi-Lo, and target-card
recurrence metrics. Specific target-card inter-arrival times such as `T:S` and
`5:S` are compared to a geometric distribution with `p = 1 / 52`. Rank-level
targets such as `T` and `5` use `p = 1 / 13`. Poisson approximations are only
appropriate for fixed-window counts, not inter-arrival gaps.

The IID game experiment runs one-box Casino Blackjack with IID cards and the
published approximate baseline strategy. It reports outcome counts, percentages,
profit path, edge per initial wager, and edge per total wager. Player blackjack
rate is calculated as natural blackjacks divided by initial hands, not dollars
wagered and not resolved hands after splits.

Win/loss streak distributions use box net result per round. Pushes are ignored
for streak continuation: `W, W, P, W` is a win streak of 3, and `L, L, P, L` is
a loss streak of 3. Signed streak plots put losses on the negative x-axis and
wins on the positive x-axis.

## Single-Box Game Validation

The single-box game validation compares six-deck `PhysicalIidCardSource` and
`One2SixCardSource` runs using identical casino blackjack rules, one shared
fixed strategy implementation, and a flat base wager. Only the card source
changes. Its primary metrics are net result, original player blackjacks, actual
double actions, and actual split actions.

Split actions are decomposed into original equal-value pair opportunities,
pair value, dealer upcard, and the fixed strategy's chosen action. Ten, jack,
queen, and king share one ten-value pair category. Under physical IID, the
theoretical original pair-opportunity probability is `25 / 169`, or about
14.7929%.

Round-profit variance and a naive round-level normal interval are recorded
without retaining every round. Those intervals are descriptive because
One2Six rounds may be serially dependent. Independent seed runs, summarized
with Student-t intervals and paired source differences, are the primary
uncertainty unit for monetary and event-rate comparisons.

Monetary streaks classify each completed round from the total result of the
active box after combining all original, split, doubled, blackjack, and other
settled hand results. A positive box result is a win, a negative result is a
loss, and zero is a push. Pushes do not count toward a streak and do not break
an open streak. Every independent seed finalizes its open streak so streaks
never continue across seed boundaries.

The signed streak histogram places loss streaks on the negative axis and win
streaks on the positive axis. Its display groups lengths beyond 20 into
explicit overflow bins, while JSON and CSV output retain every observed streak
length.

Run a smoke comparison:

```bash
python scripts/run_single_box_game_validation.py \
  --rounds 10000 \
  --base-bet 10 \
  --seeds 42,43 \
  --output-dir experiments/outputs/single_box_game_validation_split_smoke
```

Run the 20-seed replication:

```bash
python scripts/run_single_box_game_validation.py \
  --rounds 500000 \
  --base-bet 10 \
  --seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61 \
  --output-dir experiments/outputs/single_box_game_validation_20x500k
```

Run the five-seed monetary-streak validation:

```bash
python scripts/run_single_box_game_validation.py \
  --rounds 200000 \
  --base-bet 10 \
  --seeds 42,43,44,45,46 \
  --output-dir experiments/outputs/single_box_game_validation_5x200k
```

Each run writes per-seed details plus `summary.json`, `summary.csv`, and
`summary.md`, along with compact per-seed, pair decomposition, and monetary
streak CSV files plus a signed streak histogram.
This is a game-engine and source-integration validation, not an
advantage-strategy experiment.

## Six-Deck Physical IID Recurrence

`PhysicalIidCardSource` is a mathematical null model over labelled physical
cards. With six decks it has 312 stable physical IDs, but every draw is an
independent random selection from that full population. There is no depletion,
discard tray, replacement operation, cut card, shoe state, or shuffler state.
This is not a casino shoe; it is the random baseline for physical-card
recurrence comparisons.

For one specific labelled physical card in the six-deck IID model,
`p = 1 / 312`. Draw gaps follow a geometric distribution, and
`cards_between = draw_gap - 1` follows:

```text
P(cards_between = n) = (1 - p) ** n * p
```

The recurrence `metrics.json` includes the run config, target physical IDs,
plot paths, observed and theoretical tail probabilities, per-card appearance
and return-count distributions, and binned observed-vs-expected diagnostics.
The chi-square-style diagnostic is a descriptive residual summary only; it is
not currently reported as a formal p-value or significance test.

Run a smoke-sized recurrence experiment:

```bash
python scripts/run_physical_iid_recurrence_experiment.py \
  --draws 10000 \
  --deck-count 6 \
  --seed 42 \
  --output-dir experiments/outputs/physical_iid_smoke
```

Run the 1M-draw null model:

```bash
python scripts/run_physical_iid_recurrence_experiment.py \
  --draws 1000000 \
  --deck-count 6 \
  --seed 42 \
  --output-dir experiments/outputs/physical_iid_6deck_1m_seed42
```

## One2Six Recurrence Experiment

The One2Six recurrence experiment is a source-level physical-card recurrence
experiment. It does not run blackjack hands, estimate EV, optimize strategy, or
make exploitability claims. It draws from `One2SixCardSource`, returns ordered
discard batches through `accept_discards(...)`, and measures how many cards are
dealt between repeated appearances of the same `physical_id`.

The six-deck physical IID model is used as the null comparator. One2Six
recurrence is not assumed to be geometric; deviations from the physical IID
curve should be interpreted as source-mechanism diagnostics, not as evidence of
profitability. The output includes One2Six source diagnostics such as ejection
counts, fallback ejection rate, ejection group sizes, final carousel occupancy,
telemetry latency summaries, and discard-batch reappearance summaries.

Run the 1M-draw One2Six recurrence experiment:

```bash
python scripts/run_one2six_recurrence_experiment.py \
  --draws 1000000 \
  --recycle-batch-size 20 \
  --seed 42 \
  --output-dir experiments/outputs/one2six_recurrence_1m_seed42
```

## One2Six Recycle-Batch Sensitivity

The One2Six recurrence sensitivity experiment repeats the source-level
recurrence experiment across multiple `recycle_batch_size` values. This matters
because discard-return timing is a modelling assumption, and short-return
behaviour should not be treated as robust if it only appears for one batch size.

Each batch-size run writes its normal recurrence outputs into a subdirectory,
and the parent output directory receives `summary.json`, `summary.csv`,
`summary.md`, and aggregate plots. This remains source-level structure analysis,
not blackjack EV or exploitability evidence.

Run the 1M-draw sensitivity experiment:

```bash
python scripts/run_one2six_recurrence_sensitivity.py \
  --draws 1000000 \
  --recycle-batch-sizes 1,5,20,52,100 \
  --seed 42 \
  --output-dir experiments/outputs/one2six_recurrence_sensitivity_1m_seed42
```

## Multi-Box Counterfactual Action Values

The multi-box counterfactual experiment estimates the value of the next round
from a player-observable betting-boundary state. The current state is only the
composition of the visible discard rack after the preceding round has settled
and before that rack is returned during the next initial deal.

Each sampled state is cloned into branches that play one through seven boxes.
All branches begin with identical source, RNG, buffer, carousel, feeder, game,
and pending-rack state, then diverge naturally as different box counts consume
different cards. Hidden state is used only to create identical counterfactual
starting points. It is never exported as a predictor or exposed to the fixed
player strategy.

`PhysicalIidCardSource` is the negative control: visible rack composition
should not predict its next deal. The first research goal is to determine
whether any observable conditional signal appears under One2Six while staying
absent under Physical IID. This experiment does not select, optimize, or
validate a box-count strategy.

Run the smoke experiment:

```bash
python scripts/run_multi_box_counterfactual_experiment.py \
  --states-per-seed 100 \
  --seeds 42,43 \
  --burn-in-rounds 100 \
  --sample-interval-rounds 2 \
  --base-bet 10 \
  --output-dir experiments/outputs/multi_box_counterfactual_smoke
```

Run the five-seed exploratory experiment:

```bash
python scripts/run_multi_box_counterfactual_experiment.py \
  --states-per-seed 2000 \
  --seeds 42,43,44,45,46 \
  --burn-in-rounds 1000 \
  --sample-interval-rounds 5 \
  --base-bet 10 \
  --output-dir experiments/outputs/multi_box_counterfactual_5x2000
```

The compact action file contains visible rack features and branch outcomes
only. Physical IDs, source internals, RNG state, buffers, shelves, feeder
contents, and card sequences are deliberately excluded. Independent seeds are
the primary uncertainty unit.

## Observable Card-Composition Response

The observable card-response experiment measures composition directly before
using noisy blackjack profit as an endpoint. It separates two mechanisms:

1. **Immediate current-rack exclusion:** at a betting boundary, the visible
   rack remains outside the source while a cloned source emits the next 15
   cards. Fifteen positions cover the complete initial deal for seven boxes.
2. **Delayed returned-batch response:** after each observable discard batch is
   accepted, prefix sums measure future composition at exact lags 1 through 15
   and over bands extending to 1,000 dealt cards.

Low cards, neutral cards, ten-value cards, and aces remain separate. Aces are
not combined with ten-value cards because their later blackjack value depends
on whether they appear in player or dealer positions. Hi-Lo remains an
additional aggregate diagnostic.

Response slopes compare observed future composition with the simple
finite-removal benchmark implied by the visible batch. A slope near zero means
little observable finite-removal response, a slope near one resembles direct
perfectly mixed exclusion, and a negative slope indicates reversal or a
re-entry wave. These are predictive diagnostics, not isolated-batch causal
effects.

Physical IID is the negative control and should show no persistent response.
Previous unconditional monetary streak results were materially similar across
sources, so monetary streaks are not predictor features here. This experiment
is a precursor to evaluating a time-decaying machine-adjusted count, not a
completed count or betting strategy.

Run the smoke experiment:

```bash
python scripts/run_observable_card_response_experiment.py \
  --seeds 42,43 \
  --current-rack-states-per-seed 100 \
  --current-rack-burn-in-rounds 100 \
  --current-rack-sample-interval-rounds 2 \
  --current-rack-probe-cards 15 \
  --lag-rounds-per-seed 2000 \
  --lag-burn-in-rounds 100 \
  --lag-horizon-cards 1000 \
  --output-dir experiments/outputs/observable_card_response_smoke
```

Run the five-seed experiment:

```bash
python scripts/run_observable_card_response_experiment.py \
  --seeds 42,43,44,45,46 \
  --current-rack-states-per-seed 3000 \
  --current-rack-burn-in-rounds 1000 \
  --current-rack-sample-interval-rounds 5 \
  --current-rack-probe-cards 15 \
  --lag-rounds-per-seed 50000 \
  --lag-burn-in-rounds 1000 \
  --lag-horizon-cards 1000 \
  --output-dir experiments/outputs/observable_card_response_5seed
```

## Held-Out Fading-Exclusion Validation

The held-out experiment combines every active player-observable exclusion
cohort into one frozen score. Its kernel was developed on seeds 42-46 and is
validated without retuning on independent seeds 47-51:

- current visible rack: `1.00`;
- returned 1-15 dealt cards ago: `0.75`;
- returned 16-50 dealt cards ago: `0.40`;
- returned 51-100 dealt cards ago: `0.20`;
- older returned batches: `0.00`.

Each returned batch contributes to exactly one dealt-card age band. The score
is calculated before the wager and before the next initial deal, while the
current rack is still outside the source. Full current-rack exclusion therefore
applies to the initial deal; it is not assumed to apply unchanged to later hit
and dealer cards in the same round.

The primary endpoint is the next 15 cards' composition. Initial player and
dealer positions are also analysed, with aces kept separate from ten-value
cards. Flat-bet full-round profit is secondary and exploratory. Monetary
streaks are not predictors, Physical IID is the negative control, and this
phase validates a candidate signal rather than selecting a strategy.

Run a smoke validation:

```bash
python scripts/run_fading_exclusion_validation.py \
  --seeds 47,48 \
  --rounds-per-seed 2000 \
  --burn-in-rounds 100 \
  --probe-states-per-seed 100 \
  --probe-cards 15 \
  --base-bet 10 \
  --output-dir experiments/outputs/fading_exclusion_validation_smoke
```

Run the full held-out validation:

```bash
python scripts/run_fading_exclusion_validation.py \
  --seeds 47,48,49,50,51 \
  --rounds-per-seed 50000 \
  --burn-in-rounds 1000 \
  --probe-states-per-seed 3000 \
  --probe-cards 15 \
  --base-bet 10 \
  --output-dir experiments/outputs/fading_exclusion_validation_heldout
```

## Held-Out Conditional Profitability

This phase changes the primary endpoint from card composition to next-round
player profit under the unchanged source-blind fixed strategy. The development
stage uses One2Six scores from seeds 42-51 only to freeze the 10th, 30th, 70th,
and 90th percentile score-band cutpoints. It does not inspect monetary results,
game events, dealer outcomes, or validation seeds while defining those bands.

The same numerical cutpoints are then applied to fresh validation seeds 52-61
for both One2Six and the Physical IID negative control. Continuous monetary
response and conditional player edge are primary. Initial-deal composition is
retained only as a mechanism check that the frozen bands still order future
cards as expected. Monetary streaks and hidden source state are not predictors.

The fixed strategy is deliberately unchanged. Player-action optimization is a
later research phase and is justified only if observable states first show
credible conditional profitability. Such later strategy work may differ from
ordinary basic strategy because the conditional expected card distribution is
not necessarily the ordinary baseline distribution.

Run a smoke experiment:

```bash
python scripts/run_conditional_profitability_experiment.py \
  --development-seeds 42,43 \
  --validation-seeds 52,53 \
  --development-rounds-per-seed 2000 \
  --validation-rounds-per-seed 5000 \
  --burn-in-rounds 100 \
  --base-bet 10 \
  --output-dir experiments/outputs/conditional_profitability_smoke
```

Run the full held-out experiment:

```bash
python scripts/run_conditional_profitability_experiment.py \
  --development-seeds 42,43,44,45,46,47,48,49,50,51 \
  --validation-seeds 52,53,54,55,56,57,58,59,60,61 \
  --development-rounds-per-seed 20000 \
  --validation-rounds-per-seed 100000 \
  --burn-in-rounds 1000 \
  --base-bet 10 \
  --output-dir experiments/outputs/conditional_profitability_validation
```

## Score-Conditioned Player Actions

The action-value phase evaluates legal player decisions from paired copies of
the same complete game and card-source state. Natural one-box sessions continue
under the unchanged fixed strategy; at each sampled decision, every legal action
is forced once on an isolated branch and all later choices return to the fixed
strategy. No revised strategy is deployed during state generation or reporting.

Decision-time composition differs from the pre-bet score. The preceding rack
has already returned after the complete initial deal and therefore enters the
freshest returned age band. Every exposed player card across current split hands
and the dealer upcard form a new full-weight table cohort. Hit and split cards
enter that cohort before later decisions, while older returned batches age by
dealt-card count. Each observable card or batch contributes exactly once.

Low-card, ten-value, and ace shifts remain separate. Development One2Six states
on seeds 62-66 define feature-only composition cutpoints and candidate actions;
held-out seeds 67-71 cannot add or alter candidates. Generic baseline strategy
corrections are distinguished from One2Six-specific deviations. Improvements
that reduce losses in poor states remain strategically relevant even when they
do not create positive EV.

Run a smoke experiment:

```bash
python scripts/run_score_conditioned_action_values.py \
  --development-seeds 62,63 \
  --validation-seeds 67,68 \
  --decision-states-per-seed 500 \
  --burn-in-rounds 100 \
  --base-bet 10 \
  --minimum-total-state-count 20 \
  --minimum-per-seed-state-count 5 \
  --minimum-seed-sign-count 2 \
  --output-dir experiments/outputs/score_conditioned_action_values_smoke
```

Run the full experiment:

```bash
python scripts/run_score_conditioned_action_values.py \
  --development-seeds 62,63,64,65,66 \
  --validation-seeds 67,68,69,70,71 \
  --decision-states-per-seed 10000 \
  --burn-in-rounds 1000 \
  --base-bet 10 \
  --output-dir experiments/outputs/score_conditioned_action_values
```

## Monetary Streak Dependence Audit

Matching streak means or upper quantiles does not establish geometric run
lengths or independent round outcomes. The streak audit compares complete win
and loss run distributions, survival tails, and continuation probabilities
against source-and-seed geometric benchmarks. Continuation probability is the
primary shape diagnostic because an independent resolved-outcome process has a
constant continuation probability at every run length.

Pushes are removed from the resolved W/L sequence used for geometric analysis,
but they remain neutral in the live pre-bet streak: they neither increase nor
break it. A run already active after burn-in is marked left-censored, and the
final open run is right-censored. Those boundary runs are excluded from primary
PMFs while still contributing to continuation risk sets wherever continuation
is observable.

Physical IID is the required negative control. Live streak sign and length are
retained as possible strategy features only if they improve contiguous
held-out prediction beyond the unchanged frozen fading-exclusion score. This
audit does not select a betting or playing policy.

Run a smoke audit:

```bash
python scripts/run_streak_dependence_audit.py \
  --seeds 72,73 \
  --rounds-per-seed 5000 \
  --burn-in-rounds 100 \
  --base-bet 10 \
  --output-dir experiments/outputs/streak_dependence_audit_smoke
```

Run the full audit:

```bash
python scripts/run_streak_dependence_audit.py \
  --seeds 72,73,74,75,76,77,78,79,80,81 \
  --rounds-per-seed 100000 \
  --burn-in-rounds 1000 \
  --base-bet 10 \
  --output-dir experiments/outputs/streak_dependence_audit_10x100k
```

## Extreme-Tail Profitability

The strongest broad high-rich decile approached break-even, so the next test
uses frozen one-, 2.5-, five-, ten-, and twenty-percent tails to determine
whether a rarer observable state becomes positive EV under the unchanged
one-box strategy. Cutpoints use One2Six development scores only; monetary
validation outcomes cannot alter them. Physical IID remains the negative
control. Selective entry is the simplest possible advantage route, but it is
not tested as a policy unless positive conditional EV first passes the strict
held-out gate.

Insurance is audited at its actual post-initial-deal decision boundary. Because
this is a no-hole-card game, player hits, doubles, and splits can occur before
the dealer's insured second card. Ace-up insurance and even money use the exact
one-third break-even probability. Rules permit dealer-ten insurance at a
one-eleventh threshold, but operational availability is unconfirmed. Monetary
streaks are not strategy features, and this phase does not vary bets, actions,
or box count.

```bash
python scripts/run_extreme_tail_profitability.py \
  --development-seeds 82,83,84,85,86 \
  --validation-seeds 87,88,89,90,91,92,93,94,95,96 \
  --development-rounds-per-seed 20000 \
  --validation-rounds-per-seed 100000 \
  --burn-in-rounds 1000 \
  --base-bet 10 \
  --output-dir experiments/outputs/extreme_tail_profitability
```

## License

This project is licensed under the GNU General Public License v3.0 or later.

See [LICENSE](LICENSE) for the full license text.
