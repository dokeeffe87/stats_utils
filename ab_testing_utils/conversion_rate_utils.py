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
import minimum_detectable_effect_size as mdes

from time import gmtime, strftime
from tqdm import tqdm

warnings.filterwarnings('ignore')

# set the plot style
sns.set()


class ConversionExperiment:
    """
    Instantiate an experiment class. This will contain the actual experimental data, or historical data (or will generate fake data if you want).

    The distinction between a pre-experiment analysis and a post_hoc analysis with actual experiment data.

    Parameters
    -----------
    df: Optional Pandas Dataframe with input data, either historical observations, or actual experimental data. If None, will generate a fake dataset (We need to add parameters to control this)
    post_hoc: Boolean. If True, you want to do a post_hoc analysis of data
    is_experiment_data: Boolean. If True, the data represents actual experimental data.

    Examples
    ---------
    Fill this in later
    """
    def __init__(self, df: pd.DataFrame, post_hoc: bool = True, is_experiment_data: bool = True):
        self.df = df
        self.post_hoc = post_hoc
        self.is_experiment_data = is_experiment_data
        self.generate_fake_data = False

    def plot_fprs(self, fpr_dict, alpha, power):
        """
        Function to plot False Positive Risks

        :param fpr_dict: Dictionary output by the calculate_false_positive_risk function is no historical_success_rate is passed
        :param alpha: The significance level of the test (i.e. p-value threshold for declaring significance)
        :param power: The power of the experiment (1 - beta): the probability of detecting a meaningful difference between variants when there really is one. i.e. rejecting the null
                      hypothesis when there is a true difference of delta = baseline_conversion_rate * relative_minimum_detectable_effect_size
        :return: Nothing, just makes the plot.
        """

        x = np.linspace(0.05, 0.5, 100)

        plt.figure(figsize=(20, 10))
        plt.plot(x, self.calculate_false_positive_risk(alpha=alpha, power=power, historical_success_rate=x), color='blue', linewidth=3)

        for label_, success_rate in fpr_dict.items():
            plt.plot(success_rate[1], success_rate[0], 'go', markersize=12, label=label_)

        plt.title("False Positive Risk (FPR) as a function of experiment success rate at alpha={0}, beta={1}".format(alpha, 1-power), fontsize=18)
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

        se_ = sigma * np.sqrt(2/sample_size)

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

        z_alpha = stats.norm.ppf(1 - (alpha/2))
        z_power = stats.norm.ppf(power)

        n = 2 * (sigma_squared * (z_power + z_alpha)**2)/(delta_**2)

        return n

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
        v_ = (delta_/se_) - z_value

        power_ = stats.norm.cdf(x=v_)

        return power_


