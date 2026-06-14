"""Randomization-inference helpers for minimum measurable effects.

This module provides the modern import path for the legacy
`ab_testing_utils.minimum_detectable_effect_size` implementation.
"""

from ab_testing_utils.minimum_detectable_effect_size import (
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

__all__ = [
    "add_hypothetical_random_assignment",
    "calculate_difference_in_conversion",
    "compute_mme",
    "load_data",
    "make_mock_historical_data",
    "plot_simulated_null_distribution",
    "randomization_inference",
    "run",
    "save_data",
    "shift_null_distribution_by_mme",
    "visualize_mme",
]
