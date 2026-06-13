# stats_utils

A collection of statistics utilities for A/B tests and quasi-experiments.

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

Quick smoke check:

```bash
uv run python -c "from ab_testing_utils.conversion_rate_utils import ConversionExperiment; print(ConversionExperiment().calc_sample_size(power=0.8, alpha=0.05, relative_minimum_detectable_effect_size=0.1, baseline_conversion_rate=0.05))"
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
