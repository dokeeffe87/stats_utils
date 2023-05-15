"""

 _____                               _            ______      _       _   _ _   _ _
/  __ \                             (_)           | ___ \    | |     | | | | | (_) |
| /  \/ ___  _ ____   _____ _ __ ___ _  ___  _ __ | |_/ /__ _| |_ ___| | | | |_ _| |___
| |    / _ \| '_ \ \ / / _ \ '__/ __| |/ _ \| '_ \|    // _` | __/ _ \ | | | __| | / __|
| \__/\ (_) | | | \ V /  __/ |  \__ \ | (_) | | | | |\ \ (_| | ||  __/ |_| | |_| | \__ \
 \____/\___/|_| |_|\_/ \___|_|  |___/_|\___/|_| |_\_| \_\__,_|\__\___|\___/ \__|_|_|___/


A collection of utility functions for analysis of AB tests with conversion rate metrics

Author: Dan (okeeffed090@gmail.com)

V1.0.0
"""

# Import modules
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.ticker as mtick
# import minimum_detectable_effect_size as mdes

from time import gmtime, strftime
from tqdm import tqdm
from matplotlib import style

warnings.filterwarnings('ignore')

# set the plot style
style.use('fivethirtyeight')


class ConversionExperiment:
    """
    Instantiate an experiment class. This will contain the actual experimental data, or historical data (or will generate fake data if you want).

    The distinction between a pre-experiment analysis and a post_hoc analysis with actual experiment data.

    Parameters
    -----------

    Examples
    ---------
    Fill this in later
    """

    def __init__(self):
        self.generate_fake_data = False

    def plot_fprs(self, fpr_dict, alpha, power):
        """
        Function to plot False Positive Risks

        :param fpr_dict: Dictionary output by the calculate_false_positive_risk function if no historical_success_rate is passed
        :param alpha: The significance level of the test (i.e. p-value threshold for declaring significance)
        :param power: The power of the experiment (1 - beta): the probability of detecting a meaningful difference between variants when there really is one. i.e. rejecting the null
                      hypothesis when there is a true difference of delta = baseline_conversion_rate * relative_minimum_detectable_effect_size
        :return: Nothing, just makes the plot.
        """

        x = np.linspace(0.05, 0.5, 100)

        plt.figure(figsize=(20, 10))
        plt.plot(x, self.calculate_false_positive_risk(alpha=alpha, power=power, historical_success_rate=x), color='blue', linewidth=3)
        markers_ = ['go', 'rD', 'mH', 'kv']
        i = 0
        for label_, success_rate in fpr_dict.items():
            plt.plot(success_rate[1], success_rate[0], markers_[i], markersize=12, label=label_)
            i += 1

        plt.title("False Positive Risk (FPR) as a function of experiment success rate at alpha={0}, beta={1}".format(alpha, 1 - power), fontsize=18)
        plt.xlabel('Historical experiment success rate', fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel("False Positive Risk", fontsize=18)
        plt.yticks(fontsize=14)
        plt.legend()
        plt.tight_layout()
        # plt.savefig("FPR.png", facecolor='w')

    def calculate_false_positive_risk(self, alpha, power, historical_success_rate=None):
        """
        Calculates the False Positive Risk (or probability that a statistically significant result is a false positive, i.e. the probability that the null
        hypothesis is true when an experiment was statistically significant)

        :param alpha: The significance level of the test (i.e. p-value threshold for declaring significance)
        :param power: The power of the experiment (1 - beta): the probability of detecting a meaningful difference between variants when there really is one. i.e. rejecting the null
                      hypothesis when there is a true difference of delta = baseline_conversion_rate * relative_minimum_detectable_effect_size
        :param historical_success_rate: The historical rate at which A/B tests result in a true statistically significant outcome. If not provide, a range of values from the literature will
                                        be used (see Kohavi, Deng, Vermeer, 2022 for a summary of known industry values)
        :return: The False Positive Risk: i.e. the probability that the null hypothesis is true when an experiment was observed to be statistically significant
        """

        if historical_success_rate is not None:
            pi = 1 - historical_success_rate
            fpr = alpha * pi / ((alpha * pi) + (power * (1 - pi)))

            return fpr

        else:
            print('Historical success rate not provided.  Will generate False Positive Risk values for different levels of historical success')
            historical_success_rate_dict = {'Microsoft': 0.333, 'Bing': 0.15, 'Google Ads or Netflix': 0.1, 'Airbnb': 0.08}
            possible_fprs = {}
            for company_name, success_rate in historical_success_rate_dict.items():
                label_ = 'If our success rate was that of {0}'.format(company_name)
                pi_ = 1 - success_rate
                fpr_ = alpha * pi_ / ((alpha * pi_) + (power * (1 - pi_)))
                possible_fprs[label_] = (fpr_, success_rate)

            self.plot_fprs(fpr_dict=possible_fprs, alpha=alpha, power=power)

    @staticmethod
    def calc_delta(baseline_conversion_rate: float, relative_minimum_detectable_effect_size: float) -> float:
        """
        Function to compute delta parameter in experiment design.  It's just the baseline conversion rate (historical) prior to the experiment multiplied by the relative minimum
        detectable effect size

        :param baseline_conversion_rate: Expected historical conversion rate prior to experiment
        :param relative_minimum_detectable_effect_size: The minimum relative percent change in the underlying metric desired to be detected by the experiment
        :return: delta
        """

        delta_ = baseline_conversion_rate * relative_minimum_detectable_effect_size

        return delta_

    @staticmethod
    def calc_sigma(baseline_conversion_rate: float, return_square: bool = False) -> float:
        """
        Function to compute the variance of the conversion rate in the underlying population. Since the metric is a conversion type, it is modeled by a Bernoulli random variable. Hence, the
        variance is just p*(1-p), where p in this case is the baseline_conversion_rate (probability of 'success')

        :param baseline_conversion_rate: Expected historical conversion rate prior to experiment
        :param return_square: If True, returns the variance (sigma squared), if False returns the standard deviation (square root of sigma)
        :return: Either the variance or standard deviation of the underlying conversion variable
        """

        sigma_squared = baseline_conversion_rate * (1 - baseline_conversion_rate)

        if return_square:
            return sigma_squared
        else:
            return np.sqrt(sigma_squared)

    def calc_standard_error(self, sample_size: float, sigma: float = None, baseline_conversion_rate: float = None) -> float:
        """
        Function to compute the standard error in conversion rate, given a sample size and a baseline conversion rate (or pre-computed standard deviation)

        :param sample_size: The sample size determined to be needed for the experiment (this is the per-group sample size, so total sample size is twice this value)
        :param sigma: The standard deviation of the underlying conversion variable
        :param baseline_conversion_rate: Expected historical conversion rate prior to experiment
        :return: The standard error of the conversion metric
        """

        assert not all(x is not None for x in [sigma, baseline_conversion_rate]), "Either sigma or baseline_conversion_rate must be specified"

        if sigma is None:
            sigma = self.calc_sigma(baseline_conversion_rate=baseline_conversion_rate)

        se_ = sigma * np.sqrt(2 / sample_size)

        return se_

    def calc_sample_size(self, power: float, alpha: float, relative_minimum_detectable_effect_size: float, baseline_conversion_rate: float) -> float:
        """
        Function to compute the sample size (per group) of a proposed experiment on a conversion metric with baseline_conversion_rate historical conversion rate, desired relative minimum effect size
        of relative_minimum_detectable_effect_size, significance level alpha and power.  Note power here is 1 - beta

        :param power: The desired power of the experiment (1 - beta): the probability of detecting a meaningful difference between variants when there really is one. i.e. rejecting the null
                      hypothesis when there is a true difference of delta = baseline_conversion_rate * relative_minimum_detectable_effect_size
        :param alpha: The significance level of the test (i.e. p-value threshold for declaring significance)
        :param relative_minimum_detectable_effect_size: The minimum relative percent change in the underlying metric desired to be detected by the experiment
        :param baseline_conversion_rate: Expected historical conversion rate prior to experiment
        :return: The sample size required for the experiment. This is the per variant sample size, so the total sample size is twice this number
        """

        sigma_squared = self.calc_sigma(baseline_conversion_rate=baseline_conversion_rate, return_square=True)
        delta_ = self.calc_delta(baseline_conversion_rate=baseline_conversion_rate, relative_minimum_detectable_effect_size=relative_minimum_detectable_effect_size)

        z_alpha = stats.norm.ppf(1 - (alpha / 2))
        z_power = stats.norm.ppf(power)

        n = 2 * (sigma_squared * (z_power + z_alpha) ** 2) / (delta_ ** 2)

        return int(np.round(n))

    def calc_experiment_power(self, sample_size: float, alpha: float, relative_minimum_detectable_effect_size: float, baseline_conversion_rate: float, p_value: float = None) -> float:
        """
        Function to compute power of an experiment.  This should be used to calculate power of a test based on desired inputs. Note that this formulation ignores the
        possibility of type 3 error: i.e. rejection based on the wrong tail.  Can we correct for this in future versions?

        :param sample_size: The number of samples desired per variant (so this is the total experiment population / 2)
        :param alpha: The significance level of the test (i.e. p-value threshold for declaring significance)
        :param relative_minimum_detectable_effect_size: The minimum relative percent change in the underlying metric desired to be detected by the experiment
        :param baseline_conversion_rate: Expected historical conversion rate prior to experiment
        :param p_value: Observed p-value from concluded experiment. Include this only if you want to compute post-hoc power. WARNING: THIS IS AN UNRELIABLE MEASURE OF YOUR EXPERIMENTS RESULTS (See
        Kohavi, Deng, Vermeer, 2022).
        :return: The power of the test
        """
        # TODO: Add p_value functionality

        delta_ = self.calc_delta(baseline_conversion_rate=baseline_conversion_rate, relative_minimum_detectable_effect_size=relative_minimum_detectable_effect_size)
        se_ = self.calc_standard_error(sample_size=sample_size, baseline_conversion_rate=baseline_conversion_rate)

        z_value = stats.norm.ppf(1 - (alpha / 2))
        v_ = (delta_ / se_) - z_value

        power_ = stats.norm.cdf(x=v_)

        return power_

    @staticmethod
    def round_and_convert_to_int_columns(df_: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Function to format input column from DataFrame df_.  This will round numerical values and convert to integer. For experiment runtime columns, the values will be rounded up
        to the nearest integer as runtime estimates should be conservative. Caution: This will modify the input DataFrame inplace.

        :param df_: DataFrame with the column you want to format
        :param col_name: The name of the column to format
        :return: The original DataFrame with the input column now formatted.
        """

        if col_name in ['monthly_additional_conversions_upper', 'monthly_additional_conversions_lower']:
            df_[col_name] = np.round(df_[col_name])
            df_[col_name] = df_[col_name].astype(int)
        else:
            df_[col_name] = np.ceil(df_[col_name])
            df_[col_name] = df_[col_name].astype(int)

        return df_

    def create_mde_table(self, monthly_num_obs: int, baseline_conversion_rate: float, n_variants: int = 2, alpha: float = 0.05, power: float = 0.8) -> pd.DataFrame:
        """
        Function to generate a minimum detectable effect size table.  The idea here is to produce a DataFrame which matches minimum detectable effect sizes with required sample
        sizes, at the desired level of confidence and power.  Coupled with the expected monthly number of observations, this will provide an estimate for experiment runtime vs the
        size of the effect the experiment can detect.  One assumption is that the expected daily number of observations is the monthly value supplied divided by 30. It's
        important to make sure that the monthly expected number of observations is accurate, or less the experiment runtimes could be unrealistic

        :param monthly_num_obs: The expected number of observations (i.e. merchants, view, whatever the experimental unit is) seen in a 30 day period
        :param baseline_conversion_rate: Expected historical conversion rate prior to experiment
        :param n_variants: The number of variants you want to use in your test.  Usually this is two (control and exposure), but you can add any integer number here if you really
                           want to.  Be very careful when proposing multi-variant experiments! Be sure to consult with a data scientist first. Multi-variant experiments can be
                           tricky, and are prone to multiple hypothesis testing bias.
        :param alpha: The significance level of the test (i.e. p-value threshold for declaring significance)
        :param power: The desired power of the experiment (1 - beta): the probability of detecting a meaningful difference between variants when there really is one. i.e. rejecting the null
                      hypothesis when there is a true difference of delta = baseline_conversion_rate * relative_minimum_detectable_effect_size
        :return: A DataFrame with a range of different experiment runtimes, the required sample size and the magnitude of the effect that can be measured, given the supplied level
                 of confidence and power
        """

        mde_range = np.arange(0.001, 2.001, 0.001)

        sample_sizes = [self.calc_sample_size(power=power,
                                              alpha=alpha,
                                              relative_minimum_detectable_effect_size=mde,
                                              baseline_conversion_rate=baseline_conversion_rate) * n_variants for mde in mde_range
                        ]

        new_conversion_rates_upper = [baseline_conversion_rate + self.calc_delta(baseline_conversion_rate=baseline_conversion_rate,
                                                                                 relative_minimum_detectable_effect_size=mde) for mde in mde_range
                                      ]

        new_conversion_rates_lower = [baseline_conversion_rate - self.calc_delta(baseline_conversion_rate=baseline_conversion_rate,
                                                                                 relative_minimum_detectable_effect_size=mde) for mde in mde_range
                                      ]

        df_ = pd.DataFrame()
        df_['mde'] = mde_range
        df_['new_conversion_rate_upper_bound'] = new_conversion_rates_upper
        df_['new_conversion_rate_lower_bound'] = new_conversion_rates_lower
        df_['total_sample_size'] = sample_sizes
        df_['sample_size_per_variant'] = np.array(sample_sizes) / n_variants
        df_['days'] = df_['total_sample_size'] / (monthly_num_obs / 30)
        df_['weeks'] = df_['days'] / 7
        df_['monthly_additional_conversions_upper'] = (df_['new_conversion_rate_upper_bound'] * monthly_num_obs) - (baseline_conversion_rate * monthly_num_obs)
        df_['monthly_additional_conversions_lower'] = (df_['new_conversion_rate_lower_bound'] * monthly_num_obs) - (baseline_conversion_rate * monthly_num_obs)

        for col_ in ['total_sample_size', 'sample_size_per_variant', 'days', 'weeks', 'monthly_additional_conversions_upper', 'monthly_additional_conversions_lower']:
            df_ = self.round_and_convert_to_int_columns(df_=df_, col_name=col_)

        return df_

    @staticmethod
    def format_y_axis(x: float, pos) -> str:
        """
        Helper function to format the y-axis of the MDE vs runtime plot.

        :param x: A number used as a tick on a matplotlib plot that you want to format
        :param pos: This is a position indicator used internally by matplotlib to figure out where to plot the tick. You should never have to interact with it directly.
        :return: The formatted tick mark
        """

        return f"{int(x):,}"


    @staticmethod
    def plot_mde_marker(df, weeks, days, ax):
        """
        Function to format the minimum detectable effect size plot.  This will add horizontal lines to indicate the required number of weeks, as well as annotate the plot to indicate the minimum d
        etectable effect size along with the range of effect sizes that will not be detectable by the experiment at the desired level of significance and power.

        :param df: A DataFrame with experimental runtime and minimum detectable effect sizes.  This should be the output of the create_mde_table method.
        :param weeks: The number of weeks for one particular instance of the experiment, at a given level of significance, power, and effect size
        :param days: The number of days for one particular instance of the experiment, at a given level of significance, power, and effect size
        :param ax: The matplotlib ax object for the overall plot. Generated by the make_mde_plot function
        :return: Nothing. Just adds formatting to the existing plot object
        """

        ax.axhline(y=days, linestyle='--', xmax=(df[df['days'] <= days]['mde'].min() - ax.get_xlim()[0]) / ax.get_xlim()[1] - 0.005)

        if weeks > 1:
            week_text = 'weeks'
        else:
            week_text = 'week'

        ax.text(ax.get_xlim()[0], days + 1, f"{weeks} {week_text}", horizontalalignment='left')

        mde_text = "MDE = {0}%:\nConversion rates between {1}% and {2}% will not be distinguishable from baseline".format(np.round(df[df['weeks'] == weeks]['mde'].min() * 100, 3),
                                                                                                                          np.round(df[df['weeks'] == weeks]['new_conversion_rate_lower_bound'].min() * 100, 3),
                                                                                                                          np.round(df[df['weeks'] == weeks]['new_conversion_rate_upper_bound'].min() * 100, 3))

        ax.text(df[df['weeks'] <= weeks]['mde'].min() * 1.05, days - 0.5, mde_text, horizontalalignment='left')

    def make_mde_plot(self, df_, min_weeks, max_weeks, save_path=None, output_filename=None):
        """
        Function to make a plot of the minimum detectable effect sizes by experiment run time in weeks.  This is plot the number of required weeks (conservatively) against the minimum (conservatively)
        effect size detectable at the required power and significance levels

        :param df_: A DataFrame with experimental runtime and minimum detectable effect sizes.  This should be the output of the create_mde_table method.
        :param min_weeks: The minimum number of weeks you would consider running an experiment for.  It is important to take business cycles into consideration here. As a general rule, 2 weeks is a
                          bare minimum to run an A/B test
        :param max_weeks: The maximum number of weeks you would consider running an experiment for. This is a business decision; ultimately running an experiment is expensive and if the run time
                          is longer than the cost justifies, reconsider running an experiment in the first place
        :param save_path: Path to save the plot to. If None, the file will be saved to the current working directory.
        :param output_filename: Optional str name for the file where the plot will be saved. If None, the file will be called experiment_runtime_vs_mde_CURRENT_TIME.png
        :return: Nothing. Generates a plot of the minimum detectable effect sizes vs the number of required weeks at the desired level of significance and power
        """

        current_time = strftime('%Y-%m-%d_%H%M%S', gmtime())

        fig, ax = plt.subplots(figsize=(12, 8))
        df_temp = df_.copy()

        # This should remove the necessity of calculating all these mins below...
        df_temp = df_temp[['mde', 'days', 'weeks', 'new_conversion_rate_upper_bound', 'new_conversion_rate_lower_bound']].loc[(df_temp.groupby('weeks')['days'].idxmax())]

        ax.plot("mde",
                "days",
                data=df_temp.loc[(df_temp['weeks'] >= min_weeks) & (df_temp['weeks'] <= max_weeks)],
                linewidth=2,
                solid_capstyle="round",
                linestyle='--',
                marker='o',
                color='b'
        )

        ax.yaxis.set_major_formatter(mtick.FuncFormatter(self.format_y_axis))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax.set_xlabel('Minimum detectable effect size')
        ax.set_ylabel('')

        # Set limit to reasonable amount of time
        if ax.get_ylim()[1] > 60:
            ax.set_ylim([0, 7 * max_weeks * 1.2])

        # Set x-lim
        x_limit = df_temp[df_temp["weeks"] <= min_weeks]["mde"].min() * 1.2
        x_min = df_temp[df_temp['weeks'] == max_weeks]['mde'].min()
        x_min = x_min - 0.1 * x_min
        ax.set_xlim([x_min, x_limit])

        for weeks in range(min_weeks, max_weeks + 1):
            days_ = df_temp.query('weeks == {0}'.format(weeks))['days'].min()
            self.plot_mde_marker(df=df_temp, weeks=weeks, days=days_, ax=ax)

        ax.set_title('Experiment run times by minimum detectable effect size', fontsize=20)

        # We don't need the y-axis here
        ax.axes.get_yaxis().set_visible(False)

        ax.yaxis.grid(True)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if save_path is None:
            save_path = os.getcwd()

        if output_filename is None:
            file_name = 'experiment_runtime_vs_mde_{0}.png'.format(current_time)
        else:
            if not output_filename.endswith('.png'):
                file_name = output_filename + '.png'
            else:
                file_name = output_filename

        save_path = os.path.join(save_path, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
