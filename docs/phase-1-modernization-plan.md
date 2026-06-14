# Phase 1 Modernization Plan

## Context

This repo contains older hand-built statistics utilities that are still useful to humans and are now also being called by agentic skills. Before adding larger capabilities like a statistical method decision tree or causal inference workflow, Phase 1 should make the package structure safer, more discoverable, and easier for agents to use without breaking existing imports.

Current top-level packages:

- `ab_testing_utils/`
  - `conversion_rate_utils.py`
    - `ConversionExperiment`
  - `minimum_detectable_effect_size.py`
    - randomization-inference and minimum measurable effect helpers
  - `sim_utils.py`
    - `SimulateABTest`
    - `SimulateSkewedContinuous`
  - `skewed_metric_utils.py`
    - randomization-inference helpers for skewed metrics
    - `RandomizationInference`
- `quasi_experiment_utils/`
  - `interrupted_time_series_utils.py`
    - `InterruptedTimeSeries`

## Goal

Introduce a modern package namespace and documentation structure while preserving all existing behavior and import paths.

Phase 1 is intentionally conservative. It should prepare the repo for future work, not rewrite the statistical implementation itself.

## Non-goals

Phase 1 should **not** attempt to:

- redesign the A/B testing or interrupted time-series APIs;
- change statistical formulas, defaults, or plotting behavior;
- add causal inference/discovery tools;
- add the full statistical decision tree;
- remove old package names;
- force new heavy dependencies into the base install.

Those belong in later phases.

## Design principles

1. **Backward compatibility first**
   Existing company code and agent tools may import from `ab_testing_utils` or `quasi_experiment_utils`. Those imports must keep working.

2. **Additive migration**
   Add a cleaner `stats_utils` namespace, then migrate code gradually. Do not require downstream users to move immediately.

3. **Agent-friendly outputs over time**
   Future tools should expose structured result objects and markdown summaries, but Phase 1 should only establish the conventions and scaffolding.

4. **Small, reviewable changes**
   Each implementation slice should be easy to inspect and revert.

5. **Tests before behavior changes**
   Add smoke/regression tests around existing public behavior before refactoring internals.

## Proposed package layout

Introduce:

```text
stats_utils/
  __init__.py
  ab_testing/
    __init__.py
    conversion_rate.py
    minimum_detectable_effect_size.py
    simulation.py
    skewed_metrics.py
  quasi_experiments/
    __init__.py
    interrupted_time_series.py
  reporting/
    __init__.py
```

Keep existing packages as compatibility layers:

```text
ab_testing_utils/
  __init__.py
  conversion_rate_utils.py
  minimum_detectable_effect_size.py
  sim_utils.py
  skewed_metric_utils.py

quasi_experiment_utils/
  __init__.py
  interrupted_time_series_utils.py
```

### Compatibility strategy

Target end-state for Phase 1:

- New code can import from `stats_utils.*`.
- Old code can continue importing from `ab_testing_utils.*` and `quasi_experiment_utils.*`.
- Compatibility wrappers should be thin and explicit.

Example eventual wrapper pattern:

```python
# ab_testing_utils/conversion_rate_utils.py
from stats_utils.ab_testing.conversion_rate import ConversionExperiment

__all__ = ["ConversionExperiment"]
```

However, do not immediately replace old modules until tests cover current behavior. An initial safer slice can duplicate or re-export the current implementation under the new namespace first, then simplify wrappers later.

## Public API conventions for future phases

Phase 1 should establish conventions even if not all existing utilities comply yet.

New result-like objects should eventually support:

```python
result.to_dict()
result.to_markdown()
```

Future recommendation/report objects should include:

- method used or recommended;
- assumptions;
- diagnostics;
- warnings/caveats;
- interpretation guidance;
- suggested next actions.

This convention will support both human users and agentic skills.

## Implementation slices

### Slice 1: Documentation and inventory

- Add this Phase 1 plan.
- Add or update `README.md` with a short roadmap section.
- Document existing public classes/functions and their legacy import paths.

Definition of done:

- Plan exists in `docs/`.
- Current public surface is listed.
- No code behavior changed.

### Slice 2: Baseline tests

Add tests for a minimal but representative public surface:

- `ConversionExperiment.calc_delta`
- `ConversionExperiment.calc_sigma`
- `ConversionExperiment.calc_sample_size`
- one or two randomization-inference helper smoke tests with deterministic seeds/data;
- `InterruptedTimeSeries` construction and basic data-prep behavior, if feasible without brittle plotting/model-fitting checks.

Definition of done:

- `uv run pytest` passes.
- Tests document expected current behavior.
- Tests avoid long-running stochastic simulations unless seeded and small.

### Slice 3: Add `stats_utils` namespace

- Create `stats_utils/` package.
- Add subpackages for `ab_testing`, `quasi_experiments`, and `reporting`.
- Re-export existing implementation under cleaner module names.
- Update `pyproject.toml` wheel package list to include `stats_utils` while preserving old packages.

Possible first-pass mapping:

| Legacy path | New path |
| --- | --- |
| `ab_testing_utils.conversion_rate_utils` | `stats_utils.ab_testing.conversion_rate` |
| `ab_testing_utils.minimum_detectable_effect_size` | `stats_utils.ab_testing.minimum_detectable_effect_size` |
| `ab_testing_utils.sim_utils` | `stats_utils.ab_testing.simulation` |
| `ab_testing_utils.skewed_metric_utils` | `stats_utils.ab_testing.skewed_metrics` |
| `quasi_experiment_utils.interrupted_time_series_utils` | `stats_utils.quasi_experiments.interrupted_time_series` |

Definition of done:

- New imports work.
- Old imports still work.
- Tests cover both legacy and new import paths.

### Slice 4: README and examples

- Update `README.md` with:
  - package overview;
  - old-vs-new import guidance;
  - quick examples for A/B testing and interrupted time series;
  - note that causal and decision-tree tools are planned future phases.

Definition of done:

- README helps a new human or agent discover the intended API.
- Examples run or are trivial to validate.

### Slice 5: Prepare for Phase 2

- Add placeholder docs for the statistical decision tree design.
- Do **not** implement the decision tree yet unless Phase 1 tests and namespace migration are complete.

Definition of done:

- Phase 2 design questions are documented.
- Phase 1 can be reviewed independently.

## Test plan

Primary command:

```bash
uv run pytest
```

Recommended smoke checks:

```bash
uv run python -c "from ab_testing_utils.conversion_rate_utils import ConversionExperiment; print(ConversionExperiment.calc_delta(0.05, 0.1))"
uv run python -c "from stats_utils.ab_testing.conversion_rate import ConversionExperiment; print(ConversionExperiment.calc_delta(0.05, 0.1))"
```

Later, when wrappers are introduced, test that legacy and new imports refer to the same object:

```python
from ab_testing_utils.conversion_rate_utils import ConversionExperiment as LegacyConversionExperiment
from stats_utils.ab_testing.conversion_rate import ConversionExperiment as NewConversionExperiment

assert LegacyConversionExperiment is NewConversionExperiment
```

## Risks and mitigations

### Risk: breaking downstream imports

Mitigation:

- keep legacy packages in the wheel;
- add compatibility tests;
- make wrappers explicit and stable.

### Risk: accidental behavior changes during refactor

Mitigation:

- add tests before moving implementation;
- prefer re-exporting before rewriting;
- keep statistical behavior changes out of Phase 1.

### Risk: heavy dependency creep

Mitigation:

- keep Phase 1 dependency-neutral;
- reserve causal dependencies for optional extras in later phases.

### Risk: plotting-heavy APIs are brittle in automated tests

Mitigation:

- test pure calculations directly;
- smoke-test plotting/model workflows only when deterministic and fast;
- avoid snapshot-testing plots in Phase 1.

## Open questions for review

1. Should `stats_utils` become the only documented namespace immediately, or should docs present both old and new imports for a while?
2. Should compatibility wrappers emit deprecation warnings, or stay silent to avoid noisy agent/tool logs?
3. Do we want to preserve the existing module filenames exactly under `stats_utils`, or prefer cleaner names like `conversion_rate.py` and `simulation.py`?
4. Should Phase 1 include light formatting/linting configuration, or keep that separate?
5. Are there known downstream tools/imports that should be explicitly tested?

## Proposed review checklist

Before implementing code changes, review this plan for:

- compatibility assumptions;
- proposed namespace names;
- missing public APIs;
- test coverage priorities;
- whether any Phase 2 work should be pulled earlier or explicitly deferred.

## Phase 1 definition of done

Phase 1 is complete when:

- `docs/phase-1-modernization-plan.md` exists and has been reviewed;
- baseline tests cover key existing behavior;
- `stats_utils` namespace exists;
- legacy imports still work;
- package build configuration includes both old and new packages;
- README documents the modernization and future roadmap;
- no statistical behavior has intentionally changed.
