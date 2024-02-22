"""
 _____ _                         ____  ___     _        _      _   _ _   _ _
/  ___| |                       | |  \/  |    | |      (_)    | | | | | (_) |
\ `--.| | _______      _____  __| | .  . | ___| |_ _ __ _  ___| | | | |_ _| |___
 `--. \ |/ / _ \ \ /\ / / _ \/ _` | |\/| |/ _ \ __| '__| |/ __| | | | __| | / __|
/\__/ /   <  __/\ V  V /  __/ (_| | |  | |  __/ |_| |  | | (__| |_| | |_| | \__ \
\____/|_|\_\___| \_/\_/ \___|\__,_\_|  |_/\___|\__|_|  |_|\___|\___/ \__|_|_|___/


A collection of utility functions for analysis and design of experiments with highly skewed, continuous metric distributions. Essentially, whenever it's a bad idea to make
parametric assumptions.

Author: Dan (okeeffed090@gmail.com)

V1.0.0
"""

# import packages
import os
# import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
# import scipy
import math
import itertools
import warnings
import functools

from matplotlib import style
from functools import partial
from time import gmtime, strftime
from typing import Union
from tqdm import tqdm

warnings.filterwarnings('ignore')

# set the plot style
style.use('fivethirtyeight')

# TODO: Implement more pre-built test statistics (e.g. some kind of ranking metric, diversity metrics, max, min, etc).


def ri_test_statistic_difference_in_means(df: pd.DataFrame, outcome_col: str, treatment_col: str, treatment_name: str, control_name: str) -> float:
    """
    Helper function to calculate the difference in means between two groups (treatment and control). Each row of the input is meant to represent the assignment and outcome of a
    single individual.

    :param df: DataFrame with the outcome variable and treatment assignments. This assumes only two variants.
    :param outcome_col: The name of the column containing the outcome
    :param treatment_col: Name of the column containing the treatment assignments
    :param treatment_name: Name of the treatment variant contained in the treatment_col column
    :param control_name: Name of the control variant contained in the treatment_col column

    :return: Difference in means
    """

    sdo = df.query("{0}==@treatment_name".format(treatment_col))[outcome_col].mean(numeric_only=True) - df.query("{0}==@control_name".format(treatment_col))[
        outcome_col].mean(numeric_only=True)

    return sdo


def ri_test_statistic_difference_in_ks(df: pd.DataFrame, outcome_col: str, treatment_col: str, treatment_name: str, control_name: str, alternative: str = 'two-sided') -> float:
    """
    Helper function to calculate the Kolmogorov–Smirnov (KS) statistic between the outcome variable distributions of two groups (treatment and control). Each row of the input is
    meant to represent the assignment and outcome of a single individual. Use this to test if the control and treatment groups come from the same underlying distribution (or data
    generation process) or not.

    :param df: DataFrame with the outcome variable and treatment assignments. This assumes only two variants.
    :param outcome_col: The name of the column containing the outcome
    :param treatment_col: Name of the column containing the treatment assignments
    :param treatment_name: Name of the treatment variant contained in the treatment_col column
    :param control_name: Name of the control variant contained in the treatment_col column
    :param alternative: Alternative for calculating the KS statistic. Defaults to two-sided

    :return: KS statistic
    """

    ks_ = stats.ks_2samp(
        df.query("{0}==@treatment_name".format(treatment_col))[outcome_col].values,
        df.query("{0}==@control_name".format(treatment_col))[outcome_col].values, alternative=alternative)

    return ks_.statistic


def ri_test_statistic_difference_in_percentiles(df: pd.DataFrame, outcome_col: str, treatment_col: str, treatment_name: str, control_name: str, quantile: float = 0.5) -> float:
    """
    Helper function to calculate the difference between percentiles of two groups (treatment and control). Each row of the input is meant to represent the assignment and outcome of
    a single individual. Use this (with quantile = 0.5 for the median) if you're worried about outliers.

    :param df: DataFrame with the outcome variable and treatment assignments. This assumes only two variants.
    :param outcome_col: The name of the column containing the outcome
    :param treatment_col: Name of the column containing the treatment assignments
    :param treatment_name: Name of the treatment variant contained in the treatment_col column
    :param control_name: Name of the control variant contained in the treatment_col column
    :param quantile: Quantile to use for calculating the percentiles. e.g. 0.5 is the median

    :return: Difference in percentiles
    """

    q_diff = df.query("{0}==@treatment_name".format(treatment_col))[outcome_col].quantile(q=quantile) - df.query("{0}==@control_name".format(treatment_col))[
        outcome_col].quantile(q=quantile)

    return q_diff


class RandomizationInference:
    """
    Instantiate a randomization inference class.  This will contain tools for analyzing experimental data (ideally for skewed metric distributions where you'd rather not make any
    parametric assumptions), as well as experimental design for skewed continuous metrics. Right now, this only works 2 variant experimental data.

    Parameters
    -----------

    Examples
    ---------
    Fill this in later
    """

    def __init__(self):
        self._supported_test_statistics = {'difference_in_means': ri_test_statistic_difference_in_means,
                                           'difference_in_percentiles': ri_test_statistic_difference_in_percentiles,
                                           'difference_in_ks_statistic': ri_test_statistic_difference_in_ks}
        # self._supported_ci_methods = ['percentile', 'pivotal']
        self.df_sims = None
        self.observed_test_statistic = None
        self.p_val = None
        self.ci = None

    @staticmethod
    def sharp_null(df_: pd.DataFrame, sharp_null_type: str, sharp_null_value: float, outcome_column_name: str) -> pd.DataFrame:
        """
        Function to define a sharp null hypothesis for your input data. Recall that a sharp null is a null hypothesis which makes a statement about each individual unit in your
        data. The point is to be able to fill in the counterfactual outcome for each individual subject. A popular choice is Fisher's sharp null, which sets the treatment effect
        for each individual subject to 0 (i.e. the treatment does nothing at the individual treatment effect level).  The spirit of randomization inference then is to build up a
        null distribution under this sharp null hypothesis (even if it's ridiculous sounding) and test for evidence in the observed data that would let you reject this null.

        :param df_: DataFrame with the experimental data
        :param sharp_null_type: Either additive or multiplicative. Additive will add a constant value to the observed outcomes. Multiplicative will scale the observed outcomes by
                                a percentage factors.
        :param sharp_null_value: The value you want use for the sharp null. This is the value you want to use as the individual level impact of the treatment. Fisher's sharp null
                                 (i.e. no individual impact at all) would have this as zero.
        :param outcome_column_name: Name of the column in the input DataFrame which has the observed outcomes

        :return: A DataFrame with the outcomes of this assumed null hypothesis.  The values are stored in a column called outcome_sharp_null
        """
        if sharp_null_type == 'additive':
            df_['outcome_sharp_null'] = df_[outcome_column_name] + sharp_null_value
        else:
            df_['outcome_sharp_null'] = df_[outcome_column_name] * df_[outcome_column_name] * sharp_null_value

        return df_

    def select_test_statistic(self, test_statistic: dict) -> functools.partial:
        """
        Helper function to validate and set up the function used to calculate the test statistic.  This can be a custom function as long as it accepts an input dataframe and returns
        a scalar.  See the examples of pre-built test statistic functions to see how to build your own.

        :param test_statistic: This is a dictionary. The key is either a string () indicating that you want to use one of the pre-built test statistic functions, or a function that
                               accepts a dataframe and outputs a scalar. The value is another dictionary on parameters you'd like to pass to the test statistic function. You can
                               leave this as an empty dictionary {} or as None if you don't want to pass any parameters.

        :return: A partial function corresponding to the test statistic function with any passed parameters filled.
        """

        if type(test_statistic['function']) == str:
            assert test_statistic[
                       'function'] in self._supported_test_statistics, "test statistic {0} is not currently support. Please select from {1} for implement your own function".format(
                test_statistic['function'], self._supported_test_statistics)
            test_statistic['function'] = self._supported_test_statistics[test_statistic['function']]
        else:
            assert callable(test_statistic['function']), "supplied custom test statistic {0} is neither a supported type nor a function".format(test_statistic['function'])

        if test_statistic['params'] is None:
            test_statistic['params'] = {}
        else:
            assert type(test_statistic[
                'params']) == dict, "Please pass a dictionary of parameters to your test statistic function. The keys should be the parameter names and the values the parameter values."

        test_statistic_function = partial(test_statistic['function'], **test_statistic['params'])

        return test_statistic_function

    @staticmethod
    def calculate_test_statistic(df_: pd.DataFrame, test_statistic_function: functools.partial, assignments: Union[list, np.array, tuple], treatment_col: str = 'ri_in_treatment', treatment_name: Union[str, int] = 1, control_name: Union[str, int] = 0) -> float:
        """
        Function to calculate the test statistic for a given assignment

        :param df_: DataFrame containing the outcomes
        :param test_statistic_function: The function you want to use to compute the test statistic
        :param assignments: An iterable that contains the assignments for each row in the input DataFrame
        :param treatment_col: Name you want to use for the column which will contain the treatment assignments
        :param treatment_name: Name of the treatment variant in the input assignments
        :param control_name: Name of the control variant in the input assignments

        :return: The calculated test statistic
        """

        df_[treatment_col] = assignments
        # Handle numeric instability here
        stat_ = np.round(test_statistic_function(df=df_, outcome_col='outcome_sharp_null', treatment_col=treatment_col, treatment_name=treatment_name, control_name=control_name), 10)

        return stat_

    @staticmethod
    def get_all_combinations(size: int, treatment_probability: float) -> dict:
        """
        Function to calculate all possible combinations of treatment assignment. This is only really relevant when your input data is very small (small enough that it's practically
        possible to run randomization inference on all possible combinations). This should almost never happen. Note that this only works for simple coin flip (possible not 50/50)
        randomization procedures.

        :param size: Size of population sample. E.g. if you want 8 units in your experiment, this would be 8
        :param treatment_probability: The probability of being assigned to treatment. This should be the same as the probability that was actually used in the experiment.

        :return: A dictionary where the keys are the combination number (arbitrary label used later) and the values are a tuple of assignments (1 for in treatment, 0 for in
                 control)
        """

        all_combs = list(itertools.combinations(list(range(size)), int(size * treatment_probability)))
        hypothetical_assignments = {}

        for i, comb in enumerate(all_combs):
            assignment_vector = np.zeros(size)
            assignment_vector[list(comb)] = 1
            hypothetical_assignments[i] = tuple(assignment_vector)

        return hypothetical_assignments

    def make_hypothetical_assignment(self, df_: pd.DataFrame, treatment_assignment_probability: float, test_statistic_function: functools.partial, sample_with_replacement: bool, num_perms: int = 1000) -> dict:
        """
        Function to make hypothetical assignments and manage the calculation of test statistics for each assignment.  Currently, this only supports simple assignments via a
        (possibly not 50/50) coin flip.  Also, only two variants are supported at the moment. There are two supported methodologies:

        1. First, the function checks how many possible combinations of treatment assignments are possible. If that number is small enough (artificially capped at 1,000), then all
           combinations will be generated and the test statistic will be calculated for each. This really should almost never happen and is probably only relevant for very small
           datasets.

        2. Otherwise, a simulation process will be used.  If the parameter sample_with_replacement is False, the function will generate distinct hypothetical assignments (i.e.
           there will be no repeats) up to the specified number of permutations.  Otherwise, the hypothetical assignments are just sampled with replacement.

        :param df_: DataFrame with the outcomes under the sharp null
        :param treatment_assignment_probability: The probability of being assigned to the treatment group. Must be positive and less than 1.
        :param test_statistic_function: The function used to calculate the test statistic
        :param sample_with_replacement: If true, hypothetical assignment vectors will be sampled with replacement (bootstrapped). Otherwise, only distinct assignment vectors will
                                        be sampled.
        :param num_perms: Number of hypothetical assignments to draw

        :return: A dictionary. The keys label the permutation (this is an arbitrary, but unique, key).  The values are the value of the test statistic calculated for each
                 hypothetical assignment
        """

        # TODO: add support for more complex assignment strategies (e.g. blocking).

        sim_dict = {}

        # set the random seed
        np.random.seed()

        try:
            n_combs = math.comb(df_.shape[0], int(df_.shape[0] * treatment_assignment_probability))
        except ValueError:
            n_combs = np.inf

        if n_combs <= 1000:
            print('Found {0} distinct assignment combinations. All combinations will be simulated.')
            # Just get all possible assignment combinations. This should be small enough to handle in memory
            assignment_dict = self.get_all_combinations(size=df_.shape[0], treatment_probability=treatment_assignment_probability)
            for i, comb in tqdm(assignment_dict.items()):
                sim_dict[i] = self.calculate_test_statistic(df_=df_, test_statistic_function=test_statistic_function, assignments=comb)
        else:
            print('Number of distinct assignment combinations practically too large. Running {0} simulated permutations'.format(num_perms))
            if not sample_with_replacement:
                assignment_dict = {}
                i = 0
                pbar = tqdm(total=num_perms)
                while len(assignment_dict) < num_perms:  # I'm a little worried that this is going to run forever
                    assignment_tuple = tuple(stats.binom.rvs(n=1, p=treatment_assignment_probability, size=df_.shape[0]))
                    hashed_key = hash(assignment_tuple)
                    if hashed_key not in assignment_dict:
                        assignment_dict[hashed_key] = hashed_key
                        sim_dict[i] = self.calculate_test_statistic(df_=df_, test_statistic_function=test_statistic_function, assignments=assignment_tuple)
                        i += 1
                        pbar.update(1)
                pbar.close()
            else:
                for i in tqdm(range(num_perms)):
                    assignment_list = stats.binom.rvs(n=1, p=treatment_assignment_probability, size=df_.shape[0])
                    sim_dict[i] = self.calculate_test_statistic(df_=df_, test_statistic_function=test_statistic_function, assignments=assignment_list)

        return sim_dict

    def p_value(self, alternative: str) -> float:
        """
        Function to calculate the p-value of the observed test statistic given the simulated null distribution

        :param alternative: Alternative for calculating the p-value. i.e. a two-sided vs one-sided test.

        :return: The calculated p-value
        """

        if self.observed_test_statistic not in self.df_sims['test_statistic'].values:
            add_row = pd.DataFrame({"permutation": [-1], "test_statistic": self.observed_test_statistic})
            self.df_sims = pd.concat([add_row, self.df_sims])
            observed_perm = -1
        else:
            observed_perm = self.df_sims.loc[(self.df_sims['test_statistic'] == self.observed_test_statistic)]['permutation'].values[0]

        if alternative == 'two-sided':
            self.df_sims['rank_column'] = np.abs(self.df_sims['test_statistic'])
        else:
            self.df_sims['rank_column'] = self.df_sims['test_statistic']

        self.df_sims['rank'] = self.df_sims['rank_column'].rank(method='max', ascending=False)

        p_value = self.df_sims.query("permutation==@observed_perm")['rank'].values[0] / self.df_sims.shape[0]

        return p_value

    def get_ci(self, confidence: float, alternative: str = 'two-sided') -> np.array:
        """
        Function to calculate the confidence interval for the test statistic under the hypothetical null.  Only a very simple pivotal approach is supported currently.  This is
        essentially the same thing as the 'basic' method for calculating confidence intervals in the bootstrap functionality in scipy.

        :param confidence: The confidence level of the test (i.e. 0.95 for 95%)
        :param alternative: One of two-sided, less, or greater. Only two-sided is currently supported by default. Probably just remove this in future versions.

        :return: Any array with the lower and upper confidence interval bounds.
        """
        # TODO: revamp this. This doesn't really make sense actually. What we should provide is the range of value where the null hypothesis would be rejected...
        # assert method in ['percentile', 'pivotal'], "Confidence interval calculation method {0} is not supported. Currently supported methods are: {1}".format(method, self._supported_ci_methods)
        # We'll only support the basic method for now (this is actually implemented in scipy https://github.com/scipy/scipy/blob/v1.12.0/scipy/stats/_resampling.py#L279-L660)

        df_ = self.df_sims.copy()
        df_ = df_.query("permutation != -1")

        if alternative == 'two-sided':
            alpha = (1 - confidence) / 2
        else:
            alpha = 1 - confidence

        interval = alpha, 1 - alpha

        # Calculate the percentiles
        ci_low = np.percentile(df_['test_statistic'].values, interval[0] * 100)
        ci_high = np.percentile(df_['test_statistic'].values, interval[1] * 100)

        ci_low, ci_high = 2 * self.observed_test_statistic - ci_high, 2 * self.observed_test_statistic - ci_low

        if alternative == 'less':
            ci_low = np.full_like(ci_low, -np.inf)
        elif alternative == 'greater':
            ci_high = np.full_like(ci_high, np.inf)

        ci_ = np.array([ci_low, ci_high])

        return ci_

    def plot_and_output_results(self, confidence: float, alternative: str, test_stat_name: str, filename: str, output_path: str):
        """
        Function to plot results of comparison between observed test statistic and hypothetical null distribution. The output is a png file.

        :param confidence: Confidence level of the test (e.g. 0.95 for a 95% confidence level)
        :param test_stat_name: Name of the test statistic used. Currently, only works for the pre-built test statistics. Custom test statistic functions will be called
                               "custom test statistic"
        :param filename: Name of the file you want to save the output plot to.  This is optional. If you leave this out, the file name will default to
                         experimental_analysis_CURRENT_DATETIME.png
        :param output_path: Path to the directory where you want to save the output plot. If left out, this will default to the current working directory

        :return: Nothing. Generates a png file with a plot summarizing the results of the comparison
        """

        if test_stat_name == 'difference in ks statistic':
            rounding_ = 5
        else:
            rounding_ = 2

        df_for_ranking = self.df_sims.copy()
        df_for_ranking['rejection'] = df_for_ranking['rank'] / df_for_ranking.shape[0]
        rejection_edge = df_for_ranking.loc[(df_for_ranking['rejection'] <= (1 - confidence))]['rejection'].max()
        rejection_edge_values = df_for_ranking.loc[(df_for_ranking['rejection'] == rejection_edge)]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(data=self.df_sims, x='test_statistic', fill=True, ax=ax, kde=True, label='Null distribution')
        # Run a check if the there is no variation in simulated test statistics
        unique_test_stats = self.df_sims['test_statistic'].nunique()

        if unique_test_stats == 1:
            print('Warning: no variation detected in test statistic permutations. Unique value is: {0}'.format(self.df_sims['test_statistic'].unique()[0]))
        else:
            kde_x, kde_y = ax.lines[0].get_data()

            plt.axvline(x=self.observed_test_statistic, color='green', linestyle='--', label='Observed test statistic: {0}'.format(np.round(self.observed_test_statistic, rounding_)))

            if alternative == 'two-sided':
                ax.fill_between(kde_x, kde_y, where=(np.abs(kde_x) >= np.abs(rejection_edge_values['test_statistic'].max())), interpolate=True, alpha=0.5, label="Rejection region at {0}% signficance level".format(int(confidence * 100)))
            elif alternative == 'less':
                ax.fill_between(kde_x, kde_y, where=(
                            kde_x <= rejection_edge_values['test_statistic'].min()), interpolate=True, alpha=0.5, label="Rejection region at {0}% signficance level".format(int(confidence * 100)))
            elif alternative == 'greater':
                ax.fill_between(kde_x, kde_y, where=(
                            kde_x >= rejection_edge_values['test_statistic'].max()), interpolate=True, alpha=0.5, label="Rejection region at {0}% signficance level".format(int(confidence * 100)))

        ax.set_title('Distribution of test statistic under null. p-value: {0}'.format(np.round(self.p_val, 3)), fontsize=18)
        ax.set_xlabel("{0}".format(test_stat_name), fontsize=16)
        plt.legend(bbox_to_anchor=(0.7, -0.1))

        if filename is None:
            current_time = strftime('%Y-%m-%d_%H%M%S', gmtime())
            filename = 'experimental_analysis_{0}.png'.format(current_time)
        else:
            if not filename.endswith('.png'):
                filename = filename.split('.')[0] + '.png'

        if output_path is None:
            output_path = os.getcwd()

        save_path = os.path.join(output_path, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def run_randomization_inference(self, df_: pd.DataFrame, test_statistic_function: functools.partial, treatment_assignment_probability: float, sample_with_replacement: bool, num_perms: int = 1000) -> dict:
        """
        Function to handle running the randomization inference step. This just collects the results from make_hypothetical_assignments. We can probably remove this in future
        versions

        :param df_: DataFrame with the outcomes under the sharp null
        :param treatment_assignment_probability: The probability of being assigned to the treatment group. Must be positive and less than 1.
        :param test_statistic_function: The function used to calculate the test statistic
        :param sample_with_replacement: If true, hypothetical assignment vectors will be sampled with replacement (bootstrapped). Otherwise, only distinct assignment vectors will
                                        be sampled.
        :param num_perms: Number of hypothetical assignments to draw. The default is 1000, but more may be required.  Note that setting sample_with_replacement will make running
                          more permutations much faster than requiring that each be unique.  The likelihood of drawing two identical assignment vectors for a large dataset is
                          probably very small to begin with.

        :return: A dictionary. The keys label the permutation (this is an arbitrary, but unique, key).  The values are the value of the test statistic calculated for each
                 hypothetical assignment
        """

        assert 0 < treatment_assignment_probability < 1, "Treatment assignment probabilities must be great than 0 and less than 1. Received {0}".format(treatment_assignment_probability)

        sim_dict = self.make_hypothetical_assignment(df_=df_, treatment_assignment_probability=treatment_assignment_probability, test_statistic_function=test_statistic_function, num_perms=num_perms, sample_with_replacement=sample_with_replacement)

        return sim_dict

    def experimental_analysis(self, df, sharp_null_type='additive', sharp_null_value=0, test_statistic={'function': 'difference_in_means', 'params': None}, treatment_assignment_probability=0.5, outcome_column_name='y', treatment_column_name='d', treatment_name=1, control_name=0, num_permutations=1000, alternative='two-sided', confidence=0.95, sample_with_replacement=False, filename=None, output_path=None):
        """
        Function to handle running randomization inference on a two variant AB test. The point is to use randomization inference to test the hypothesis of the experiment.  The
        function here breaks up the randomization inference process into 4 steps:

        1. Implement a selected sharp null hypothesis

        2. Pick a test statistic. We have a list of pre-built ones, otherwise a function must be supplied. This function must consume a DataFrame and return a single scalar value

        3. Select a randomization procedure. Currently, this only supports random assignment via a (possible non 50/50) coin flip

        4. Calculate the test p-value

        The final results are output as a figure which plots the hypothetical null distribution for the test statistic, the p-value of the test, and the rejection region.

        :param df: DataFrame contain unit assignments and observed outcomes from a two variant experiment. Each row should correspond to the outcome for a unique unit
        :param sharp_null_type: Either additive or multiplicative. Additive will add a constant value to the observed outcomes. Multiplicative will scale the observed outcomes by
                                a percentage factors.
        :param sharp_null_value: The value you want use for the sharp null. This is the value you want to use as the individual level impact of the treatment. Fisher's sharp null
                                 (i.e. no individual impact at all) would have this as zero.
        :param test_statistic: This is a dictionary. The key is either a string () indicating that you want to use one of the pre-built test statistic functions, or a function that
                               accepts a dataframe and outputs a scalar. The value is another dictionary on parameters you'd like to pass to the test statistic function. You can
                               leave this as an empty dictionary {} or as None if you don't want to pass any parameters.
        :param treatment_assignment_probability: The probability of being assigned to the treatment group. Must be positive and less than 1.
        :param outcome_column_name: Name of the column in the input DataFrame which has the observed outcomes
        :param treatment_column_name: Name of the column in input DataFrame which contains the treatment assignments
        :param treatment_name: Name of the treatment variant in the input DataFrame
        :param control_name: Name of the control variant in the input DataFrame
        :param num_permutations: Number of hypothetical assignments to draw. The default is 1000, but more may be required.  Note that setting sample_with_replacement will make
                                 running more permutations much faster than requiring that each be unique.  The likelihood of drawing two identical assignment vectors for a large
                                 dataset is probably very small to begin with.
        :param alternative: Alternative for calculating the p-value. i.e. a two-sided vs one-sided test.
        :param confidence: Confidence level of the test (e.g. 0.95 for a 95% confidence level)
        :param sample_with_replacement: If true, hypothetical assignment vectors will be sampled with replacement (bootstrapped). Otherwise, only distinct assignment vectors will
                                        be sampled.
        :param filename: Name of the file you want to save the output plot to.  This is optional. If you leave this out, the file name will default to
                         experimental_analysis_CURRENT_DATETIME.png
        :param output_path: Path to the directory where you want to save the output plot. If left out, this will default to the current working directory

        :return: Nothing. Generates a png file with a plot summarizing the results of the comparison
        """

        assert sharp_null_type in ['additive', 'multiplicative'], "only additive or multiplicative sharp nulls are supported. Received {0}".format(sharp_null_type)

        assert type(num_permutations) == int, "Only an integer number of permutations is possible. Received {0}".format(num_permutations)

        assert alternative in ['two-sided', 'less', 'greater'], "Only {0} alternatives are supported. Received {0}".format(alternative)

        # Copy the input DataFrame so that we don't modify the original data in_place
        df_ = df.copy()

        # Save the input test statistic name if it's a string
        if type(test_statistic['function']) == str:
            test_stat_name = test_statistic['function'].replace('_', ' ')
        else:
            test_stat_name = 'custom test statistic'

        # Step 1: implement the selected sharp null
        df_ = self.sharp_null(df_=df_, sharp_null_type=sharp_null_type, sharp_null_value=sharp_null_value, outcome_column_name=outcome_column_name)

        # Step 2: pick a test statistic. We have a list of pre-built ones, otherwise a function must be supplied
        # This function must consume a DataFrame and return a single scalar value
        test_statistic_function = self.select_test_statistic(test_statistic=test_statistic)

        # Step 3: Select a randomization procedure.
        # Currently, this only supports random assignment via coin flip. The probability of being in the treatment group is customizable
        # Also run the simulations
        # print('Running {0} permutations'.format(num_permutations))
        print('Running randomization inference...')
        sim_dict = self.run_randomization_inference(df_=df_, test_statistic_function=test_statistic_function, treatment_assignment_probability=treatment_assignment_probability, num_perms=num_permutations, sample_with_replacement=sample_with_replacement)

        self.df_sims = pd.DataFrame.from_dict(sim_dict, orient='index')
        self.df_sims = self.df_sims.reset_index()
        self.df_sims.columns = ['permutation', 'test_statistic']

        # Step 4: calculate p-values
        # First, calculate the observed statistic difference
        self.observed_test_statistic = test_statistic_function(df=df_, outcome_col=outcome_column_name, treatment_col=treatment_column_name, treatment_name=treatment_name, control_name=control_name)

        self.p_val = self.p_value(alternative=alternative)

        # final output. This should be summarized in a plot
        self.plot_and_output_results(confidence=confidence, alternative=alternative, test_stat_name=test_stat_name, filename=filename, output_path=output_path)
