"""
 _____ _           _   _ _   _ _
/  ___(_)         | | | | | (_) |
\ `--. _ _ __ ___ | | | | |_ _| |___
 `--. \ | '_ ` _ \| | | | __| | / __|
/\__/ / | | | | | | |_| | |_| | \__ \
\____/|_|_| |_| |_|\___/ \__|_|_|___/


Helper functions for simulating experiment results for testing and benchmarking

Author: Dan (okeeffed090@gmail.com)

V1.0.0
"""

# Import modules
import os
import sys
import pandas as pd
import numpy as np

from typing import Union


class SimulateABTest:
    @staticmethod
    def simulate_expected_daily_visitors(number_of_days_for_experiment: int, daily_num_observations: int) -> pd.DataFrame:
        """
        Static method for simulating a count of daily experimental units for the AB test.  This uses the desired number of days for the simulated experiment and the expected daily average
        number of units that should qualify for the experiment to simulate a daily count of eligible units.  The count is assigned via a Poisson distribution. The returned DataFrame has two
        columns, one labels the days of the simulated experiment, and the other is just an integer 1, indicating a unique unit.  Grouping by the day column and summing the units column will
        give the total number of simulated experimental units observed per day

        :param number_of_days_for_experiment: Integer number of days you want to simulate the experiment for
        :param daily_num_observations: Integer average number of expected visitors per day
        :return: A DataFrame which labels each day in the experiment and a 1 for each unit observed on that day.

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

    @staticmethod
    def assign_randomly(df: pd.DataFrame, n_variants: int = 2, p_vals: Union[list, str, np.ndarray] = 'equal', group_col: str = 'group') -> pd.DataFrame:
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
            assert type(p_vals) in [list, np.ndarray], "if p_vals is not 'equal', then it must be either a list or np.ndarray"
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

    @staticmethod
    def generate_conversions(df: pd.DataFrame, conversion_rate_dict: dict, group_col: str = 'group') -> pd.DataFrame:
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

    def run_sim(self, daily_num_observations: int, number_of_days_for_experiment: int, expected_conversion_rates: Union[list, np.ndarray], group_col: str, p_vals: Union[list, str, np.ndarray] = 'equal') -> pd.DataFrame:
        """

        :param daily_num_observations:
        :param number_of_days_for_experiment:
        :param expected_conversion_rates:
        :param group_col:
        :param p_vals:
        :return:

        """

        # Calculate derived values from inputs
        n_variants = len(expected_conversion_rates)

        if type(p_vals) == str:
            assert p_vals == 'equal', 'The only accepted string value for this parameter is equal'
        else:
            assert type(p_vals) in [list, np.ndarray], "if p_vals is not 'equal', then it must be either a list or np.ndarray"
            assert sum(p_vals) == 1, "probabilities of assignment to each group {0} must sum to 1".format(p_vals)
            assert len(p_vals) == len(expected_conversion_rates), "There must be the same number of assignment probabilities as number of expected conversion rates"

        assert type(expected_conversion_rates) in [list, np.ndarray], "must be either a list or np.ndarray"

        # Generate simulated daily observations
        df_ab = self.simulate_expected_daily_visitors(number_of_days_for_experiment=number_of_days_for_experiment, daily_num_observations=daily_num_observations)

        df_exp = self.assign_randomly(df=df_ab, n_variants=n_variants, p_vals=p_vals, group_col=group_col)

        # check summary for assignments
        print("Simulated group sizes: \n")
        print(df_exp[group_col].value_counts(normalize=True))

        # Check daily summary stats for assignments
        print("\n")
        print("Simulated daily group sizes: \n")
        print(df_exp[['day', group_col]].groupby('day').value_counts(normalize=True))
        print("\n")

        # Generate group to conversion rate
        simulate_conversion_rates_dict = {}
        for g_ in df_exp[group_col].unique():
            if g_ == 'control':
                group_index = 0
            else:
                group_index = int(g_.split('_')[-1])
            conv_ = expected_conversion_rates[group_index]
            simulate_conversion_rates_dict[g_] = conv_

        df_exp = self.generate_conversions(df=df_exp, conversion_rate_dict=simulate_conversion_rates_dict, group_col=group_col)

        print("\n")
        print("Simulated conversion rates by group: \n")
        print(df_exp[[group_col, 'conversion']].groupby(group_col).value_counts(normalize=True))
        print("\n")

        return df_exp




