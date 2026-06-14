import numpy as np
import pandas as pd

from stats_utils.ab_testing.minimum_detectable_effect_size import (
    calculate_difference_in_conversion,
    compute_mme,
    make_mock_historical_data,
    shift_null_distribution_by_mme,
)


def test_make_mock_historical_data_has_expected_shape_and_columns():
    df = make_mock_historical_data(number_of_data_points=25, null_conversion_rate=0.2)

    assert list(df.columns) == ["data_unit_id", "data_unit_converted"]
    assert df.shape == (25, 2)
    assert df["data_unit_id"].tolist() == list(range(1, 26))
    assert set(df["data_unit_converted"].unique()).issubset({0, 1})


def test_calculate_difference_in_conversion_uses_treatment_minus_control():
    df = pd.DataFrame(
        {
            "in_treatment": [0, 0, 1, 1],
            "converted": [0, 1, 1, 1],
        }
    )

    assert calculate_difference_in_conversion(df, data_unit_col_name="converted") == 0.5


def test_compute_mme_uses_lower_tail_implied_by_power():
    simulated_null = [0.0, 1.0, 2.0, 3.0]

    assert compute_mme(
        critical_value=2.0,
        simulated_effects_under_null_distribution=simulated_null,
        power=0.5,
    ) == 0.5


def test_shift_null_distribution_by_mme_adds_constant_shift():
    shifted = shift_null_distribution_by_mme(
        simulated_effects_under_null_distribution=[0.0, 1.0, 2.0],
        mme=0.25,
    )

    assert np.allclose(shifted, [0.25, 1.25, 2.25])
