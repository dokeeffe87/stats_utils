# stats_utils

A collection of hand-built statistics utilities for A/B tests and quasi-experiments.

This project is being modernized so it can serve both humans doing statistical analysis and agentic tools that need stable, inspectable statistical workflows.

## Current capabilities

- A/B test utilities for conversion-rate metrics.
- Minimum measurable effect helpers using randomization inference.
- Simulation helpers for A/B tests and skewed continuous metrics.
- Randomization-inference helpers for highly skewed/non-parametric experiment metrics.
- Interrupted time-series helpers for quasi-experiments.

## Import paths

New code should prefer the modern `stats_utils` namespace:

```python
from stats_utils.ab_testing.conversion_rate import ConversionExperiment
from stats_utils.ab_testing.skewed_metrics import RandomizationInference
from stats_utils.quasi_experiments.interrupted_time_series import InterruptedTimeSeries
```

Legacy imports remain supported for backwards compatibility:

```python
from ab_testing_utils.conversion_rate_utils import ConversionExperiment
from ab_testing_utils.skewed_metric_utils import RandomizationInference
from quasi_experiment_utils.interrupted_time_series_utils import InterruptedTimeSeries
```

See `docs/public-api-inventory.md` for the full legacy-to-modern mapping.

## Quick smoke check

```bash
uv run python -c "from stats_utils.ab_testing.conversion_rate import ConversionExperiment; print(ConversionExperiment().calc_sample_size(power=0.8, alpha=0.05, relative_minimum_detectable_effect_size=0.1, baseline_conversion_rate=0.05))"
```

Expected output:

```text
29826
```

## Roadmap

The modernization plan is tracked in `docs/phase-1-modernization-plan.md`.

Planned future phases:

1. **Phase 1: package modernization and compatibility**
   - Add the `stats_utils` namespace.
   - Preserve legacy imports.
   - Add baseline tests and documentation.
2. **Phase 2: statistical decision tree**
   - Recommend statistical methods from structured problem descriptions.
   - Return assumptions, diagnostics, alternatives, caveats, and reporting guidance.
3. **Phase 3+: causal workflow tools**
   - Make causal assumptions explicit with DAGs.
   - Identify estimands and adjustment sets.
   - Add estimator/refutation/reporting workflows with strong communication of limitations.

## Setup with uv

This project is configured as a Python package managed by [uv](https://docs.astral.sh/uv/). The checked-in `.python-version` pins the default interpreter to Python 3.11, which is a safer target for the older scientific/statistics dependencies in this repo.

From a fresh clone:

```bash
uv sync
```

That will create a local `.venv/`, install the package in editable mode, and install the dependencies declared in `pyproject.toml`.

To run Python inside the environment:

```bash
uv run python
```

## Tests

```bash
uv run pytest
```

## Notebooks

Notebook dependencies are optional. Install them with:

```bash
uv sync --extra notebooks
```

Then start Jupyter with:

```bash
uv run jupyter lab
```

Or register the environment as a notebook kernel:

```bash
uv run python -m ipykernel install --user --name stats-utils --display-name "Python (stats-utils)"
```

## Common commands

```bash
# Install/update the environment from the lockfile
uv sync

# Add a new runtime dependency
uv add package-name

# Add a notebook-only dependency
uv add --optional notebooks package-name

# Run a module/script with the managed environment
uv run python path/to/script.py
```
