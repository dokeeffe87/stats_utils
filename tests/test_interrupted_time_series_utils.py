import pandas as pd
import pytest

from stats_utils.quasi_experiments.interrupted_time_series import InterruptedTimeSeries


def test_prep_data_for_interrupted_time_series_adds_design_columns_in_date_order():
    treatment_date = pd.Timestamp("2024-01-03")
    its = InterruptedTimeSeries(treatment_date=treatment_date)
    its.date_col = "date"

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "outcome": [10, 20, 30, 40],
        }
    )

    prepped = its.prep_data_for_interrupted_time_series(df)

    assert prepped["date"].tolist() == list(pd.date_range("2024-01-01", periods=4))
    assert prepped["T"].tolist() == [1, 2, 3, 4]
    assert prepped["D"].tolist() == [0, 0, 1, 1]
    assert prepped["P"].tolist() == [0, 0, 0, 1]


def test_prep_data_for_interrupted_time_series_rejects_duplicate_dates():
    treatment_date = pd.Timestamp("2024-01-03")
    its = InterruptedTimeSeries(treatment_date=treatment_date)
    its.date_col = "date"

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "outcome": [10, 20],
        }
    )

    with pytest.raises(AssertionError, match="Each row must be a unique date"):
        its.prep_data_for_interrupted_time_series(df)
