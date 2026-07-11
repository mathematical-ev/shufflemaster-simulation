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
- Use the same fixed strategy implementation for card-source comparisons.
- Card-source experiments must not branch player decisions by source type,
  source state, physical identity, telemetry, or prior-card history.
- Test rule correctness with deterministic card sequences rather than random
  simulations.
- The current casino blackjack rule profile stands on soft 17.
- Do not interpret one random monetary result as evidence of an advantage.
- Do not add venue-specific or personal identifying names to tracked files.
- Use multiple independent seeds before interpreting small monetary source
  differences.
- Distinguish initial equal-value pair opportunities from actual split actions.
- Use seed-level replication as the primary uncertainty unit when a source can
  create serial dependence between rounds.
- Do not use naive round-IID or binomial assumptions as the sole evidence for a
  One2Six source difference.
- Preserve generic public terminology and repository privacy rules in reports
  and generated labels.
- Define a round-level monetary streak from the total settled net result of the
  entire active box, after combining all original and split hands.
- Classify positive box net as a win, negative box net as a loss, and zero as a
  push.
- Pushes do not count toward and do not break an open win or loss streak.
- Finalize every open streak at an independent seed boundary; never join
  streaks across seeds.
- Use negative signed lengths for loss streaks, positive signed lengths for win
  streaks, and omit zero from streak distributions.
- Do not optimize box count before demonstrating an observable out-of-sample
  signal.
- Use identical complete starting states for counterfactual action comparisons.
- Never expose physical IDs, RNG state, buffers, shelves, feeder contents, or
  other internal source state to player-facing features or strategies.
- Use independent seeds as the primary uncertainty unit for counterfactual
  state-action estimates.
- Use Physical IID as the negative control for observable-state experiments.
- Do not interpret one favourable state bucket as an advantage strategy.
- Keep experiment runtime proportional to the live research question.
- Separate current-rack exclusion from returned-batch re-entry response.
- Use player-observable composition features only; hidden source state may
  establish identical probes but must never enter predictors or exports.
- Preserve ace and ten-value response categories separately.
- Test observable card composition before returning to blackjack EV endpoints.
- Use independent seeds as the primary uncertainty unit for response slopes.
- Do not optimize box count before a stable observable response kernel exists.
- Do not infer that a globally stationary source is IID at every conditional
  state.
- Do not retune frozen fading-exclusion weights on held-out seeds.
- Use new independent seeds for held-out signal validation and retain Physical
  IID as a required null control.
- Combine all overlapping active observable cohorts into one state before any
  strategy or EV optimization work; each returned batch belongs to one
  dealt-card age band only.
- Distinguish out-of-sample signal validation from strategy and EV
  optimization.
- Maintain player/dealer and ace/ten-value asymmetry in initial-deal analyses.
- Keep experiment runtime proportional to the current research question.
- Stop adding composition diagnostics once an observable signal is sufficiently
  validated; translate validated signals into held-out conditional EV.
- Do not optimize action decisions before establishing state-conditioned
  profitability under the unchanged source-blind baseline strategy.
- Define score bands from score distributions only, without monetary outcomes,
  and freeze them before processing fresh independent validation seeds.
- Require Physical IID as the negative control for conditional-profitability
  experiments.
- Preserve strategy/source blindness during baseline EV evaluation.
- Keep composition metrics only as mechanism checks once monetary EV becomes
  the primary endpoint.
- Optimize player decisions across both favourable and unfavourable observable
  states; loss reduction is useful even when a state remains negative EV.
- Calculate composition at the actual player-decision boundary, after the prior
  rack return and with all exposed table cards included exactly once.
- Keep low-card, ten-value, and ace effects separate in action-value work.
- Estimate legal actions with paired counterfactual branches from identical
  complete game, source, and RNG states.
- Distinguish generic baseline strategy corrections from source-specific
  composition deviations.
- Do not deploy a revised strategy before candidate discovery and independent
  held-out validation are complete.
- Do not infer resolved-outcome independence from matching streak means or
  quantiles; compare complete run distributions, survival, and continuation
  hazards against source-specific geometric benchmarks.
- Preserve canonical monetary streak semantics: aggregate the complete box net,
  treat positive as win and negative as loss, and let pushes neither increment
  nor break the current streak.
- Keep boundary-censored runs out of primary geometric PMFs and include them in
  continuation risk sets only where continuation is observable.
- Test whether streak state adds held-out predictive value beyond the frozen
  fading-exclusion score before retaining it as a strategy feature.
- Use independent-seed uncertainty and paired One2Six-minus-Physical-IID source
  comparisons for streak-shape and predictive-value claims.
- Keep streak-audit runtime proportional to the live advantage question.
- Prioritise held-out conditional EV over additional descriptive diagnostics.
- Test frozen extreme score tails before adding betting, action, or box-count
  complexity; define cutpoints without monetary outcomes.
- Retain Physical IID as the negative control for every extreme-tail claim.
- Do not assume a rules-permitted optional wager is operationally available.
- Require a robust seed-level margin over exact insurance and even-money
  break-even thresholds before treating them as feasible.
- Keep extreme-tail and insurance-audit runtime proportional to the live
  advantage question.

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
