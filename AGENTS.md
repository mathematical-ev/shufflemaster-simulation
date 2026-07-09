# AGENTS.md

## Repository Purpose

`shufflemaster-sim` simulates blackjack-like card games to test whether
continuous shuffling machines, especially Shuffle Master / One2Six-style
devices, can create exploitable short-horizon structure.

The first milestone is a clean, testable Python project with explicit
boundaries between card generation, game rules, player decisions, validation,
and settlement. Do not expand into full casino game logic until rules are
specified and tests exist.

## Architecture Boundaries

- Keep card sources separate from game rules.
- Keep game rules separate from player strategies.
- Keep player strategy separate from legal-action validation.
- Keep settlement/accounting separate from card generation.
- Keep actions explicit and testable.
- Do not use uncontrolled randomness in game logic.
- Inject `random.Random` instances or seeds anywhere randomness is required.
- Prefer small pure functions and typed dataclasses.
- House rules must be implemented from explicit source material, not memory.
- The published strategy is an approximate baseline only.
- The current runnable card sources are IID random, generic finite shoe,
  manual shoe, and a configurable One2Six-style source model.
- Ordered discard-rack collection and shuffling-device return timing are
  explicitly modelled because they will matter for One2Six simulation.
- Physical card identity is critical for One2Six return-time analysis.
- Do not replace stable `physical_id` with draw-event `draw_id`.
- Card sources own card generation, shoe construction, shuffling, and return
  mechanics.
- Game rules must not depend on how a card source works internally.
- Manual shoe and One2Six must remain separate concepts.
- Do not model a continuous shuffler as a finite shoe.
- Treat `One2SixCardSource` defaults as explicit modelling assumptions, not
  claims about proprietary device internals.
- Preserve One2Six telemetry and invariant checks when changing source logic.
- Accepted discard order matters for return-time analysis; do not reorder cards
  unless a tested card-source rule explicitly does so.
- Future multi-box work must allow independent bet amounts per box.
- Per-box results and streaks must remain independent.
- Pushes do not break win/loss streaks; treat them as neutral ignored outcomes
  for streak continuation.
- Keep experiment framework code separate from the core engine.
- Core modules under `src/shufflemaster_sim/` must not import experiment
  modules.
- Inter-arrival time under IID should be compared to a geometric distribution.
- Poisson approximations are suitable for fixed-window counts, not
  inter-arrival time.
- Always validate the IID baseline before interpreting One2Six output.
- Physical IID is a mathematical null model, not a manual shoe and not
  One2Six.
- Physical recurrence experiments for six-deck comparisons should use 312
  labelled physical cards.
- Do not confuse symbol/rank IID recurrence with labelled physical-card IID
  recurrence.
- Physical IID recurrence outputs must include config metadata for
  reproducibility.
- Do not report formal statistical significance unless a proper test and its
  assumptions are implemented.
- Diagnostic residuals are not exploitability evidence by themselves.
- Run and inspect the physical IID recurrence baseline before interpreting
  One2Six recurrence output.
- One2Six recurrence outputs must include source diagnostics such as ejection
  counts, fallback rates, group sizes, occupancy, and available latency
  summaries.
- Do not interpret recurrence deviations as exploitability without game-level
  and strategy-level evidence.
- Before treating One2Six recurrence patterns as robust, run sensitivity over
  recycle batch size.
- Do not treat one recycle batch size as robust evidence.
- Keep source-level recurrence structure separate from game-level
  exploitability analysis.

## Important Directories

- `src/shufflemaster_sim/`: importable package code.
- `tests/`: pytest tests.
- `.vscode/`: tracked workspace settings for the local `.venv`.
- `scripts/`: small CLI entry points only; the engine belongs in `src/`.
- `experiments/`: experiment metrics, plotting, runners, and generated-output
  handling outside the core engine.

## Setup, Test, and Lint Commands

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

## Coding Standards

- Use type hints for public functions and classes.
- Keep modules PEP8 compliant and formatted with Ruff.
- Add or update tests when behavior changes.
- Prefer protocols for replaceable simulation components.
- Keep runtime dependencies minimal.

## License policy

Repository source code is licensed under GPL-3.0-or-later.

New Python source files should include:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Roudenko
```

Generated experiment outputs under `experiments/outputs/` are not source code
and should remain untracked unless explicitly needed.

Do not copy incompatible third-party code into this repository.

## Do-Not Rules

- Do not add heavy dependencies without justification.
- Do not extend One2Six / Shuffle Master logic beyond explicitly requested and
  tested source-model assumptions.
- Do not implement Casino Blackjack, Pontoon, or other casino rules from memory.
  Leave placeholders until rules are explicitly specified.
- Do not make claims about exploitability without simulation evidence.
- Do not treat one run's profit as evidence of exploitability.
- Future work should include a solver-generated exact house-rule strategy.
- Future work should validate and calibrate One2Six source assumptions before
  making any exploitability claims.
