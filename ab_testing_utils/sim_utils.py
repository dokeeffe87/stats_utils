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
import scipy.stats as stats

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
        Function to simulate random assignment to n variant test.  Just uses a simple multinomial distribution to generate assignment vectors

        :param df: DataFrame with historical (or simulated) assignments. Expected that each row represents on unit. Assignments will be stored in a DataFrame column called
                   group_col (input variable).  By default, the "0" group will be control. All others will be labeled as treatment_i, where i represents the index past 0 in the
                   p_vals iterable.

        :param n_variants: Number of variants to assign
        :param p_vals: Assignment probabilities to each variant. Must be iterable, or 'equal'. 'equal' just assumes that each variant has equal probability of assignment
        :param group_col: Name for the column you want to contain the assignments in the output

        :return: DataFrame with the variant assignments in the group_col column
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
        Generates hypothetical conversion events. Conversion events are modeled by a binomial distribution.

        :param df: DataFrame where each row represents a unit. Must be labeled by assignment group
        :param conversion_rate_dict: A dictionary which contains the hypothetical conversion probabilities for each variant in the input DataFrame. The keys should be the variant
                                     names, and the values the conversion probabilities
        :param group_col: Name of the column in the input DataFrame which contains the group assignments

        :return: A copy of the input DataFrame with a column called conversion. This will be a binary 0 or 1 representing non conversion or conversion, respectively.
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
        Helper function to run through each of the steps for generating a simulated conversion experiment

        :param daily_num_observations: Integer average number of expected visitors per day
        :param number_of_days_for_experiment: Integer number of days you want to simulate the experiment for
        :param expected_conversion_rates: Iterable of expected conversion rates per variant. It's assumed that the first element is the expected conversion rate of the control group
        :param p_vals: Assignment probabilities to each variant. Must be iterable, or 'equal'. 'equal' just assumes that each variant has equal probability of assignment
        :param group_col: Name for the column you want to contain the assignments in the output

        :return: DataFrame where each row represents an experiment unit. Contains assignment date, variant name, and whether the unit converted.

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


class SimulateSkewedContinuous(SimulateABTest):
    # Just use the simulate_expected_daily_visitors and assign_randomly methods from the SimulateAB class
    # Add support for dropout, model continuous outcomes with gamma, zero inflation as well

    def simulate_skewed_experiment(self, daily_num_observations: int, number_of_days_for_experiment: int, group_col: str, p_vals: Union[list, str, np.ndarray] = 'equal', zero_skewed_outcome=False, a_params: tuple[float] = (0.01, 0.05), scale_params: tuple[float] = (10000, 11000), outcome_col_name: str = 'outcome', rounding: int = 3, loc_params: tuple[float] = (0, 0.1)) -> pd.DataFrame:
        """

        :param daily_num_observations: Integer number of days you want to simulate the experiment for
        :param number_of_days_for_experiment: Integer average number of expected visitors per day
        :param group_col: Name for the column you want to contain the assignments in the output
        :param p_vals: Assignment probabilities to each variant. Must be iterable, or 'equal'. 'equal' just assumes that each variant has equal probability of assignment
        :param zero_skewed_outcome: Boolean. If True, will simulate a zero inflated outcome as a Gamma distribution. Otherwise, a normal distribution will be used
        :param a_params: Tuple of a parameters for the rvs method of the Gamma distribution in scipy.stats. First value is assumed to be for the control group, the remainder for treatment groups
                         Ignored if zero_skewed_outcome = False
        :param scale_params: Tuple of scale parameters for the rvs method of the Gamma distribution in scipy.stats or the norm distribution in scipy.stats. First value is assumed to be for the
                             control group, the remainder for treatment groups
        :param outcome_col_name: Name you want for the output column containing the simulated values
        :param rounding: The degree of rounding to apply.  If the continuous outcome is meant to be a dollar value, for example, you should set this to 2
        :param loc_params: Tuple of loc parameters for the rvs method of the norm distribution in scipy.stats. First value is assumed to be for the
                           control group, the remainder for treatment groups Ignored if zero_skewed_outcome = True
        :return:
        """

        # Calculate derived values from inputs
        n_variants = len(scale_params)

        if type(p_vals) == str:
            assert p_vals == 'equal', 'The only accepted string value for this parameter is equal'
        else:
            assert type(p_vals) in [list, np.ndarray], "if p_vals is not 'equal', then it must be either a list or np.ndarray"
            assert sum(p_vals) == 1, "probabilities of assignment to each group {0} must sum to 1".format(p_vals)
            assert len(p_vals) == len(scale_params), "There must be the same number of assignment probabilities as number of distribution parameters"

        df_sim = self.simulate_expected_daily_visitors(number_of_days_for_experiment=number_of_days_for_experiment, daily_num_observations=daily_num_observations)
        df_exp = self.assign_randomly(df=df_sim, n_variants=n_variants, p_vals=p_vals, group_col=group_col)

        # check summary for assignments
        print("Simulated group sizes: \n")
        print(df_exp[group_col].value_counts(normalize=True))

        # Check daily summary stats for assignments
        print("\n")
        print("Simulated daily group sizes: \n")
        print(df_exp[['day', group_col]].groupby('day').value_counts(normalize=True))
        print("\n")

        outcomes_ = []
        for g_ in df_exp[group_col].unique():
            if g_ == 'control':
                group_index = 0
            else:
                group_index = int(g_.split('_')[-1])
            if zero_skewed_outcome:
                df_ = self.simulate_zero_skewed_outcomes(df=df_exp.query("{0}=='{1}'".format(group_col, g_)),
                                                         a=a_params[group_index],
                                                         scale=scale_params[group_index],
                                                         outcome_col_name=outcome_col_name,
                                                         rounding=rounding)
            else:
                df_ = self.simulate_normal_distributed_outcomes(df=df_exp.query("{0}=='{1}'".format(group_col, g_)),
                                                                loc=loc_params[group_index],
                                                                scale=scale_params[group_index],
                                                                outcome_col_name=outcome_col_name,
                                                                rounding=rounding)
            outcomes_.append(df_)

        df_simulated = pd.concat(outcomes_)

        return df_simulated

    @staticmethod
    def simulate_normal_distributed_outcomes(df: pd.DataFrame, loc: float = 0, scale: float = 1, outcome_col_name: str = 'outcome', rounding: int = 3) -> pd.DataFrame:
        """
        Function to simulate a continuous outcome which follows a normal distribution

        :param df: DataFrame containing a list of units you want to simulate an outcome for.  Each row must be a uniquely observed unit
        :param loc: loc parameter for the rvs method of the norm distribution in scipy.stats
        :param scale: scale parameter for the rvs method of the norm distribution in scipy.stats
        :param outcome_col_name: Name you want for the output column containing the simulated values
        :param rounding: The degree of rounding to apply.  If the continuous outcome is meant to be a dollar value, for example, you should set this to 2

        :return: The original input DataFrame with the additional outcome_col_name column containing simulated values
        """
        # TODO: Merge this with the simulate_zero_skewed_outcomes function. Doesn't really make sense to have separate functions to do this
        outcomes_ = []
        df_ = df.copy()

        for d_ in df_['day'].unique():
            num_samples = df_.query("day=='{0}'".format(d_))['units'].sum()
            r = list(stats.norm.rvs(size=num_samples, loc=loc, scale=scale))
            outcomes_ = outcomes_ + r

        df_[outcome_col_name] = outcomes_
        df_[outcome_col_name] = df_[outcome_col_name].apply(lambda x: np.round(x, rounding))

        return df_

    @staticmethod
    def simulate_zero_skewed_outcomes(df: pd.DataFrame, a: float = 0.01, scale: int = 10000, outcome_col_name: str = 'outcome', rounding: int = 3) -> pd.DataFrame:
        """
        Function to simulated highly skewed continuous outcomes on input data

        :param df: DataFrame containing a list of units you want to simulate an outcome for.  Each row must be a uniquely observed unit
        :param a: a parameter for the rvs method of the Gamma distribution in scipy.stats
        :param scale: scale parameter for the rvs method of the Gamma distribution in scipy.stats
        :param outcome_col_name: Name you want for the output column containing the simulated values
        :param rounding: The degree of rounding to apply.  If the continuous outcome is meant to be a dollar value, for example, you should set this to 2

        :return: The original input DataFrame with the additional outcome_col_name column containing simulated values
        """

        outcomes_ = []
        df_ = df.copy()

        for d_ in df_['day'].unique():
            num_samples = df_.query("day=='{0}'".format(d_))['units'].sum()
            r = list(stats.gamma.rvs(size=num_samples, a=a, scale=scale))
            outcomes_ = outcomes_ + r

        df_[outcome_col_name] = outcomes_
        df_[outcome_col_name] = df_[outcome_col_name].apply(lambda x: np.round(x, rounding))

        return df_

    @staticmethod
    def adjust_for_dropout(df: pd.DataFrame, dropout_prob: float, outcome_col_name: str = 'outcome', adjust_dropout_prob: bool = True):
        """

        :param df:
        :param dropout_prob:
        :param outcome_col_name:
        :param adjust_dropout_prob:
        :return:
        """

        df_zeros = df.query("@outcome_col_name==0")
        df_not_zeros = df.query("@outcome_col_name > 0")

        if adjust_dropout_prob:
            dropout_prob = dropout_prob * df.shape[0] / df_zeros.shape[0]

        dropout_labels = np.random.binomial(n=1, p=dropout_prob, size=df_zeros.shape[0])

        df_zeros['is_dropout'] = dropout_labels
        df_not_zeros['is_dropout'] = 0

        df_with_do = pd.concat([df_zeros, df_not_zeros])

        return df_with_do
