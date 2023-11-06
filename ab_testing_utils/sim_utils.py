"""

 _____                               _            ______      _       _   _ _   _ _
/  __ \                             (_)           | ___ \    | |     | | | | | (_) |
| /  \/ ___  _ ____   _____ _ __ ___ _  ___  _ __ | |_/ /__ _| |_ ___| | | | |_ _| |___
| |    / _ \| '_ \ \ / / _ \ '__/ __| |/ _ \| '_ \|    // _` | __/ _ \ | | | __| | / __|
| \__/\ (_) | | | \ V /  __/ |  \__ \ | (_) | | | | |\ \ (_| | ||  __/ |_| | |_| | \__ \
 \____/\___/|_| |_|\_/ \___|_|  |___/_|\___/|_| |_\_| \_\__,_|\__\___|\___/ \__|_|_|___/


Helper functions for simulating experiment results for testing and benchmarking

Author: Dan (okeeffed090@gmail.com)

V1.0.0
"""

# Import modules
import os
import sys
import pandas as pd
import numpy as np


def simulate_expected_daily_visitors(number_of_days_for_experiment, daily_num_observations):
    """

    :param number_of_days_for_experiment:
    :param daily_num_observations:
    :return:
    """

    daily_units = []
    day_list = []
    for i in range(number_of_days_for_experiment):
        day_ = str(i)
        number_of_observations = np.random.poisson(daily_num_observations)
        observations = [1] * number_of_observations
        daily_units = daily_units + observations
        day_index = [day_] * number_of_observations
        day_list = day_list + day_index

    df_ = pd.DataFrame()
    df_['day'] = day_list
    df_['units'] = daily_units

    return df_


def assign_randomly(df, n_variants=2, p_vals='equal', group_col='group'):
    """

    :param df:
    :param n_variants:
    :param p_vals:
    :param group_col:
    :return:
    """

    df_ = df.copy()

    if p_vals == 'equal':
        p_vals = [1.0 / n_variants] * n_variants
    else:
        assert type(p_vals) in [list, np.ndarray], "if p_vals is not 'equal', then it must be either a list of np.ndarray"
        assert sum(p_vals) == 1, "probabilities of assignment to each group {0} must sum to 1".format(p_vals)

    df_g = df_.groupby('day', as_index=False).sum()
    assignment = []
    for day_, num_obs in zip(df_g['day'].values, df_g['units'].values):
        assignments_ = np.random.multinomial(n=1, pvals=p_vals, size=num_obs)
        assignments_ = [np.argmax(x) for x in assignments_]
        assignment = assignment + list(assignments_)
    df_[group_col] = assignment
    df_[group_col] = df_[group_col].apply(lambda x: 'control' if x == 0 else 'treatment_{0}'.format(x))

    return df_


def generate_conversions(df, conversion_rate_dict, group_col='group'):
    """

    :param df:
    :param conversion_rate_dict:
    :param group_col:
    :return:
    """

    df_ = df.copy()
    df_cr = pd.DataFrame.from_dict(conversion_rate_dict, orient='index').reset_index()
    df_cr.columns = [group_col, 'conversion_probability']

    df_ = df_.merge(df_cr, on=group_col, how='left')

    assert not df_cr.isnull().values.any(), 'Ensure that all treatment groups are represented in the input conversion_rate_dict'

    df_['conversion'] = df_['conversion_probability'].apply(lambda x: np.random.binomial(n=1, p=x))

    return df_


def run_sim(daily_num_observations, number_of_weeks_for_experiment, n_variants, p_vals, expected_conversion_rates, group_col):

    assert len(p_vals) == len(expected_conversion_rates), "There must be the same number of assignment probabilities as number of expected conversion rates"
    assert len(p_vals) == n_variants, "There must be one assignment probability per variant"
    assert len(expected_conversion_rates) == n_variants, "There must be one expected conversion rate per variant"

    # need: daily_num_observations, baseline_conversion_rate, number_of_weeks_for_experiment, n_variants, p_vals, group_col,
    # TODO: Maybe just accept a dictionary of group name --> probability of assignment?

    # Calculate derived values from inputs
    n_variants = len(p_vals)

    monthly_num_observations = daily_num_observations * 7 * 4
    number_of_days_for_experiment = number_of_weeks_for_experiment * 7

    # Generate simulated daily observations
    df_ab = pd.DataFrame()
    day_list, daily_units = simulate_expected_daily_visitors(number_of_days_for_experiment=number_of_days_for_experiment, daily_num_observations=daily_num_observations)
    df_ab['day'] = day_list
    df_ab['units'] = daily_units

    df_exp = assign_randomly(df=df_ab, n_variants=n_variants, p_vals=p_vals)

    # check summary for assignments
    print("Simulated group sizes: \n")
    print(df_exp[group_col].value_counts(normalize=True))

    # Check daily summary stats for assignments
    print("\n")
    print("Simulated daily group sizes: \n")
    print(df_exp[['day', group_col]].groupby('day').value_counts(normalize=True))
    print("\n")

    # Generate group to conversion rate dict




