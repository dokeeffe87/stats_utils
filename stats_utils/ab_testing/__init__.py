"""A/B testing utilities under the modern `stats_utils` namespace."""

from stats_utils.ab_testing.conversion_rate import ConversionExperiment
from stats_utils.ab_testing.minimum_detectable_effect_size import (
    add_hypothetical_random_assignment,
    calculate_difference_in_conversion,
    compute_mme,
    load_data,
    make_mock_historical_data,
    plot_simulated_null_distribution,
    randomization_inference,
    run,
    save_data,
    shift_null_distribution_by_mme,
    visualize_mme,
)
from stats_utils.ab_testing.simulation import SimulateABTest, SimulateSkewedContinuous
from stats_utils.ab_testing.skewed_metrics import (
    RandomizationInference,
    ri_test_statistic_difference_in_ks,
    ri_test_statistic_difference_in_means,
    ri_test_statistic_difference_in_percentiles,
)

__all__ = [
    "ConversionExperiment",
    "RandomizationInference",
    "SimulateABTest",
    "SimulateSkewedContinuous",
    "add_hypothetical_random_assignment",
    "calculate_difference_in_conversion",
    "compute_mme",
    "load_data",
    "make_mock_historical_data",
    "plot_simulated_null_distribution",
    "randomization_inference",
    "ri_test_statistic_difference_in_ks",
    "ri_test_statistic_difference_in_means",
    "ri_test_statistic_difference_in_percentiles",
    "run",
    "save_data",
    "shift_null_distribution_by_mme",
    "visualize_mme",
]
