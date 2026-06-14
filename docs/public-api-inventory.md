# Public API Inventory

This inventory captures the current public surface before Phase 1 modernization. The legacy paths listed here should continue to work after the `stats_utils` namespace is introduced.

## Legacy A/B testing package

Package: `ab_testing_utils`

### `ab_testing_utils.conversion_rate_utils`

- `ConversionExperiment`
  - Conversion-rate A/B test design and analysis helpers.
  - Includes sample-size, standard-error, MDE, false-positive-risk, plotting, and post-hoc analysis utilities.

### `ab_testing_utils.minimum_detectable_effect_size`

- `make_mock_historical_data`
- `load_data`
- `save_data`
- `add_hypothetical_random_assignment`
- `calculate_difference_in_conversion`
- `randomization_inference`
- `plot_simulated_null_distribution`
- `compute_mme`
- `shift_null_distribution_by_mme`
- `visualize_mme`
- `run`

### `ab_testing_utils.sim_utils`

- `SimulateABTest`
  - Simulates daily visitors, assignment, and conversion outcomes.
- `SimulateSkewedContinuous`
  - Early helpers for simulating skewed continuous metric outcomes.

### `ab_testing_utils.skewed_metric_utils`

- `ri_test_statistic_difference_in_means`
- `ri_test_statistic_difference_in_ks`
- `ri_test_statistic_difference_in_percentiles`
- `RandomizationInference`
  - Randomization-inference workflow for skewed/non-parametric experiment metrics.

## Legacy quasi-experiment package

Package: `quasi_experiment_utils`

### `quasi_experiment_utils.interrupted_time_series_utils`

- `InterruptedTimeSeries`
  - Interrupted time-series workflow with OLS, naive ARIMA, and auto-ARIMA model options.

## Modern namespace mapping

| Legacy path | Modern path |
| --- | --- |
| `ab_testing_utils.conversion_rate_utils` | `stats_utils.ab_testing.conversion_rate` |
| `ab_testing_utils.minimum_detectable_effect_size` | `stats_utils.ab_testing.minimum_detectable_effect_size` |
| `ab_testing_utils.sim_utils` | `stats_utils.ab_testing.simulation` |
| `ab_testing_utils.skewed_metric_utils` | `stats_utils.ab_testing.skewed_metrics` |
| `quasi_experiment_utils.interrupted_time_series_utils` | `stats_utils.quasi_experiments.interrupted_time_series` |

## Compatibility policy for Phase 1

- Legacy imports remain supported.
- Modern imports are additive.
- No deprecation warnings are emitted in Phase 1 to avoid noisy downstream agent/tool logs.
- Tests should verify that representative legacy and modern imports resolve to the same objects.
