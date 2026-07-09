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

`PublishedApproxCasinoStrategy` is a starting published H17 multi-deck basic
strategy constrained by Casino Blackjack legal actions. It is not yet a
solver-generated exact Casino Blackjack strategy.

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
