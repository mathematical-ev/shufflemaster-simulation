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

## Star Blackjack Baseline

Run the one-box IID Star Blackjack baseline simulation:

```bash
python scripts/run_star_blackjack_baseline.py --rounds 10000 --base-bet 10 --seed 42 --card-source iid
```

Run the same baseline with a finite shuffled shoe:

```bash
python scripts/run_star_blackjack_baseline.py --rounds 10000 --base-bet 10 --seed 42 --card-source finite-shoe --deck-count 6
```

Run the Star-style manual shoe benchmark:

```bash
python scripts/run_star_blackjack_baseline.py --rounds 10000 --base-bet 10 --seed 42 --card-source manual-shoe --deck-count 8 --cut-card-penetration 0.75
```

The baseline uses `IidRandomCardSource` and a configurable base bet. It is
structured so the card source can later be replaced by a finite shoe or a
One2Six / Shuffle Master simulator.

`PublishedApproxStarStrategy` is a starting published H17 multi-deck basic
strategy constrained by Star Blackjack legal actions. It is not yet a
solver-generated exact Star Blackjack strategy.

The engine explicitly models ordered discard-rack collection and
shuffling-device return timing. Current-round discards are staged for return
only after the next round's initial deal, which matters for future One2Six-style
card-source simulation.

The current runnable simulation supports IID random cards, a generic
finite-shoe source, and a Star-style manual shoe benchmark. It does not make
exploitability claims; those require future card-source models and simulation
evidence.

The current CLI still runs one box, but internally boxes are represented with
independent wagers such as `box_bets = {1: 10.0}`. Per-box win/loss streaks are
tracked by each box's net result per round. A neutral/push round resets both
current streaks.

## Card Sources And Identity

`IidRandomCardSource` treats every draw as a new physical card. It is useful for
baseline game-mechanics validation, but it is not a shoe or shuffler model.

`FiniteShoeCardSource` builds a finite set of physical cards, shuffles them, and
draws without replacement. Accepted discards are stored in order and are not
drawable again until the shoe is empty and the explicit reshuffle policy returns
the discard tray to the shoe.

`ManualShoeCardSource` is the Star-style card-shoe benchmark. It defaults to
eight decks and a configurable 75% cut-card penetration assumption. The cut-card
threshold marks a shuffle for the next round; the source does not reshuffle in
the middle of a round. On shuffle, remaining shoe cards and accepted discards
are combined while preserving physical card identities.

Cards carry two identities:

- `physical_id`: stable identity for the real card moving through shoe, hand,
  discard rack, shuffler, and possible later redraw.
- `draw_id`: unique identity for the specific draw event.
