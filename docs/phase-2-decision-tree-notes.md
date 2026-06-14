# Phase 2 Notes: Statistical Decision Tree

Phase 2 should build a method-selection layer for humans and agents. This file intentionally captures design questions only; implementation should wait until Phase 1 modernization is complete.

## Working goal

Create an inspectable, structured recommendation engine that maps a statistical problem description to ranked statistical methods, assumptions, diagnostics, caveats, and reporting guidance.

## Candidate core objects

```python
StatProblem(
    goal="compare_groups",
    outcome_type="continuous",
    exposure_type="categorical",
    n_groups=2,
    paired=False,
    repeated_measures=False,
    randomized=True,
    hierarchical=False,
    time_series=False,
    sample_size="medium",
    distribution_notes=["skewed"],
)
```

```python
MethodRecommendation(
    primary="Welch's t-test",
    alternatives=["Mann-Whitney U", "permutation test", "bootstrap CI"],
    assumptions=[...],
    diagnostics=[...],
    avoid_if=[...],
    reporting_guidance="...",
)
```

## Design questions

1. Should rules live in Python, YAML, or both?
2. Should the first version focus on experiment analysis and group comparisons before expanding to regression, survival, causal, and time-series methods?
3. What schema should agent tools pass in when users ask vague statistical questions?
4. How should recommendations express uncertainty when required context is missing?
5. Should the decision tree ask clarifying questions, or always return best-effort recommendations with missing-information warnings?

## Suggested first implementation slice

- Define `StatProblem` and `MethodRecommendation` dataclasses.
- Implement 10–20 high-confidence rules for common group-comparison and experiment-analysis problems.
- Add `to_dict()` and `to_markdown()` outputs.
- Add tests for method routing and warning behavior.
