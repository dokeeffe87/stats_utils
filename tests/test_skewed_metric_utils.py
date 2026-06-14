import numpy as np
import pandas as pd

from stats_utils.ab_testing.skewed_metrics import (
    RandomizationInference,
    ri_test_statistic_difference_in_ks,
    ri_test_statistic_difference_in_means,
    ri_test_statistic_difference_in_percentiles,
)


def _example_experiment_df():
    return pd.DataFrame(
        {
            "variant": ["control", "control", "treatment", "treatment"],
            "outcome": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_difference_in_means_uses_treatment_minus_control():
    assert ri_test_statistic_difference_in_means(
        _example_experiment_df(),
        outcome_col="outcome",
        treatment_col="variant",
        treatment_name="treatment",
        control_name="control",
    ) == 2.0


def test_difference_in_percentiles_uses_treatment_minus_control():
    assert ri_test_statistic_difference_in_percentiles(
        _example_experiment_df(),
        outcome_col="outcome",
        treatment_col="variant",
        treatment_name="treatment",
        control_name="control",
        quantile=0.5,
    ) == 2.0


def test_difference_in_ks_returns_two_sample_ks_statistic():
    assert ri_test_statistic_difference_in_ks(
        _example_experiment_df(),
        outcome_col="outcome",
        treatment_col="variant",
        treatment_name="treatment",
        control_name="control",
    ) == 1.0


def test_randomization_inference_selects_builtin_test_statistic():
    ri = RandomizationInference()

    statistic = ri.select_test_statistic(
        {
            "function": "difference_in_means",
            "params": {
                "treatment_name": 1,
                "control_name": 0,
            },
        }
    )

    df = pd.DataFrame({"outcome_sharp_null": [1.0, 2.0, 3.0, 4.0]})
    assignments = np.array([0, 0, 1, 1])

    assert ri.calculate_test_statistic(
        df_=df,
        test_statistic_function=statistic,
        assignments=assignments,
    ) == 2.0


def test_get_all_combinations_returns_complete_small_assignment_space():
    combinations = RandomizationInference.get_all_combinations(size=4, treatment_probability=0.5)

    assert len(combinations) == 6
    assert all(len(assignment) == 4 for assignment in combinations.values())
    assert all(sum(assignment) == 2 for assignment in combinations.values())
