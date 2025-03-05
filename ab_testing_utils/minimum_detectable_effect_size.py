"""
___  ______  ___ _____   _____       _            _       _
|  \/  ||  \/  ||  ___| /  __ \     | |          | |     | |
| .  . || .  . || |__   | /  \/ __ _| | ___ _   _| | __ _| |_ ___  _ __
| |\/| || |\/| ||  __|  | |    / _` | |/ __| | | | |/ _` | __/ _ \| '__|
| |  | || |  | || |___  | \__/\ (_| | | (__| |_| | | (_| | || (_) | |
\_|  |_/\_|  |_/\____/   \____/\__,_|_|\___|\__,_|_|\__,_|\__\___/|_|


Author: Dan (dan.okeeffe@shopify.com)

Use historical data to calculate a minimum measurable effect size for use in online A/B tests, given a desired power.  This is done through randomization inference.

Largely adapted from: https://medium.com/towards-data-science/practical-experiment-fundamentals-all-data-scientists-should-know-f11c77fea1b2
"""

# import modules
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats as stats

from multiprocessing import Pool
from time import strftime, gmtime
from pathlib import Path

# set plot style
sns.set()


def make_mock_historical_data(number_of_data_points: int = 1000, null_conversion_rate: float = 0.01) -> pd.DataFrame:
    """
    Function to build a mock historical dataset as a stand-in for conversion under the null distribution.  You should really only use this for testing. A real experiment should be
    based on real historical data

    :param number_of_data_points: The number of data points you want to generate in the mock dataset
    :param null_conversion_rate: The conversion rate you expect under the null hypothesis
    :return: A DataFrame with the mock historical data.  It will have a column called data_unit_id which is the unique identifier value for each row (for example, this could be
             shop_id), and a binary column called data_unit_converted.  A 1 in the data_unit_converted column means that the data unit in that row converted. Zero means they did
             not convert
    """
    data_point_ids = list(range(1, number_of_data_points+1))
    # We will assign to two groups (this only works for straight A/B testing at this point)
    data_unit_converted = stats.binom.rvs(n=1, p=null_conversion_rate, size=number_of_data_points)
    df_ = pd.DataFrame({'data_unit_id': data_point_ids, 'data_unit_converted': data_unit_converted})

    return df_


def load_data(path_to_data: str) -> pd.DataFrame:
    """
    Function to read the historical data

    :param path_to_data: The path to the dataset you want to load.  Currently, only supports simple csv files
    :return: The input data as a pandas DataFrame
    """

    df = pd.read_csv(path_to_data)

    return df


def save_data():
    pass


def add_hypothetical_random_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function will add a hypothetical column in_treatment which will record is a data unit is in the control or treatment groups

    :param df: Input DataFrame with conversion data (binary) with one row per data unit
    :return: The input DataFrame with an extra column in_treatment which denotes if the data unit is in the treatment or not (1 for yes, 0 for no)
    """

    # Make a copy of the input DataFrame in order to avoid inadvertently modifying it in place
    df_ = df.copy()

    scipy.random.seed()

    # This will add a random assignment to each row of the input DataFrame with probability 0.5 (i.e. equally likely to be in control or treatment groups)
    df_['in_treatment'] = stats.binom.rvs(n=1, p=0.5, size=df_.shape[0])

    return df_


def calculate_difference_in_conversion(df: pd.DataFrame, data_unit_col_name: str) -> float:
    """
    This function will group by the hypothetical random treatment assignments and compute the mean difference between the treatment and control groups, which is the test statistic

    :param df: DataFrame with the conversion data and the hypothetical random assignments
    :param data_unit_col_name: The name of the column which contains the conversion state of each unit in the data
    :return: The effect size which is the difference in the mean conversion from the hypothetical treatment group and the hypothetical control group
    """

    # Make a copy of the input DataFrame in order to not inadvertently modify it in place
    df_ = df.copy()
    df_agg = pd.DataFrame(df_.groupby(['in_treatment'], as_index=False)[data_unit_col_name].mean()).pivot_table(columns=['in_treatment'], values=[data_unit_col_name])

    effect_size = df_agg[1].values[0] - df_agg[0].values[0]

    return effect_size


def randomization_inference(df: pd.DataFrame, data_unit_col_name, number_of_simulations: int = 10000, run_in_parallel: bool = True, random_seed: int = None) -> list:
    """
    This function will run the actual randomization inference simulation.  It takes in a DataFrame containing conversion and data unit id data, and will repeat the random
    assignment + effect size calculation for number_of_simulations times.  This will build up the distribution of the effect size under the null hypothesis.
    :param df: DataFrame with binary conversion data per data unit
    :param data_unit_col_name: The name of the column which contains the conversion state of each unit in the data
    :param number_of_simulations: The number of randomization inference simulations to run in total
    :param run_in_parallel: Boolean. If True, will run the simulation process in parallel
    :param random_seed: Random seed for reproducibility
    :return: A list of values which represent the distribution of effect size under the null distribution
    """

    # Make a copy of the input DataFrame in order to not inadvertently modify it in place
    df_ = df.copy()

    if random_seed is not None:
        np.random.seed(random_seed)

    if run_in_parallel:
        workers = os.cpu_count()
        with Pool(workers) as p:
            assignment_result_list_of_dfs = p.map(add_hypothetical_random_assignment, [df_]*number_of_simulations)
            # TODO: Fix this
            effect_size_cal = p.map(calculate_difference_in_conversion, assignment_result_list_of_dfs, [data_unit_col_name]*number_of_simulations)
    else:
        assignment_result_list_of_dfs = []
        effect_size_cal = []
        for i in range(number_of_simulations):
            assignment_result_list_of_dfs.append(add_hypothetical_random_assignment(df_))
            effect_size_cal.append(calculate_difference_in_conversion(df=assignment_result_list_of_dfs[-1], data_unit_col_name=data_unit_col_name))

    return effect_size_cal


def plot_simulated_null_distribution(simulated_effect_sizes_ri: list, critical_value: float, current_time: str, p_value: float = None, height: int = 10,
                                     significance_level: float = 0.05, aspect: float = 1.5, save_directory: str = None):

    # Generate the save name for the figure
    file_name = 'simulated_null_distribution_{0}.png'.format(current_time)
    if save_directory is not None:
        # save_directory = save_directory + 'figures/'
        file_name = os.path.join(save_directory, file_name)

    plt.figure()
    g = sns.displot(simulated_effect_sizes_ri, rug=True, kind='kde', height=height, aspect=aspect, fill=False)
    if p_value is None:
        plt.title('Simulated distribution of difference in sample conversions: randomization inference', fontsize=18)
    else:
        plt.title('Randomization Inference null distribution with computed p-value: {0}'.format(np.round(p_value, 4)), fontsize=18)
    plt.ylabel('Density', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlabel('Simulated difference in sample conversion', fontsize=16)
    plt.xticks(fontsize=14)

    g.fig.set_figwidth(15)
    g.fig.set_figheight(10)
    plt.axvline(critical_value, color='red', linestyle='--', linewidth=6)
    if p_value is not None:
        plt.axvline(p_value, color='green', linestyle='--', linewidth=6)

    # Label the two regions of interest as well as the critical value line
    # Make the plot locations less heuristic.
    area_labels = ['  Effect distribution \nunder null hypothesis', '{}%'.format(np.round(significance_level*100), 2)]
    for ax in g.axes.flat:
        print(ax)
        y_vals = ax.get_yticks()
        y_diff = y_vals[1] - y_vals[0]
        y_loc = y_diff / 2.0
        ax.text(critical_value + 0.2*critical_value, y_loc, area_labels[1], fontsize=16)
        ax.text(critical_value - 1.3*critical_value, y_loc, area_labels[0], fontsize=16)
        plt.text(critical_value + 0.05*critical_value,
                 np.median(y_vals),
                 "Critical value: {0}".format(np.round(critical_value, 4)),
                 rotation=90,
                 verticalalignment='center',
                 fontsize=16)
        if p_value is not None:
            plt.text(p_value + 0.05*p_value,
                     np.median(y_vals),
                     "P-value: {0}".format(np.round(p_value, 4)),
                     rotation=90,
                     verticalalignment='center',
                     fontsize=16)

    # Fill in the area of significance with a different color
    for ax in g.axes.flat:
        kde_x, kde_y = ax.lines[0].get_data()
        if p_value is None:
            ax.fill_between(kde_x, kde_y, where=(kde_x >= critical_value), interpolate=True, color='red', alpha=0.3)
            ax.fill_between(kde_x, kde_y, where=(kde_x <= critical_value), interpolate=True, color='blue', alpha=0.3)

    plt.tight_layout()
    plt.savefig(file_name)


def compute_mme(critical_value: float, simulated_effects_under_null_distribution: list, power: float = 0.8) -> float:
    """
    This function will compute the minimum measurable effect.  It takes the simulated effect size values under the null distribution, then computes the 1 - power level
    percentile.  This means that if the power of the test we want to build is 80%, then this will compute the 20th percentile of effect sizes under the null. The mme
    is then the critical value (computed elsewhere) minus the percentile computed here.  Effectively, we assume that the true effect size is constant, and then shift the
    distribution of effect sizes under the null by this value.  For a power of 80% this would mean shifting the null distribution until 80% of the area under it is beyond the
    critical region.  The amount we need to shift is the mme.  This is because it is the smallest true difference in conversion that we would be able to get a statistically
    significant result with at least 80% probability.

    :param critical_value:
    :param simulated_effects_under_null_distribution:
    :param power:
    :return:
    """

    q_ = 100 - power*100
    simulated_effect_size_qth_percentile = np.percentile(simulated_effects_under_null_distribution, q_)
    mme = critical_value - simulated_effect_size_qth_percentile

    return mme


def shift_null_distribution_by_mme(simulated_effects_under_null_distribution: list, mme: float) -> list:
    """
    This function will shift the simulated effect size distribution under the null by the calculated mme

    :param simulated_effects_under_null_distribution: List of effect sizes simulated from the null distribution
    :param mme: The minimum measurable effect size calculated in the compute_mme function
    :return: A list of values which are the simulated effects under the null distribution shifted by the mme (a constant)
    """

    minimum_measurable_effect_ri = np.array(simulated_effects_under_null_distribution) + mme

    return list(minimum_measurable_effect_ri)


def visualize_mme(simulated_effect_sizes_ri: list, minimum_measurable_effect_ri: list, power: float, critical_point_ri: float, mme: float, current_time: str,
                  significance_level: float = 0.05, save_directory: str = None):
    """

    :param simulated_effect_sizes_ri:
    :param minimum_measurable_effect_ri:
    :param power:
    :param critical_point_ri:
    :param mme:
    :param current_time:
    :param significance_level:
    :param save_directory:
    :return:
    """
    # Generate the save name for the figure
    file_name = 'null_distribution_and_mme_shifted_distribution_{0}.png'.format(current_time)
    if save_directory is not None:
        # save_directory = save_directory + 'figures/'
        file_name = os.path.join(save_directory, file_name)

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.kdeplot(simulated_effect_sizes_ri, fill=False, ax=ax, color='black')
    sns.kdeplot(minimum_measurable_effect_ri, fill=False, ax=ax, color='blue')
    plt.xlabel('Difference in conversion', fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel('Density', fontsize=16)
    plt.yticks(fontsize=14)
    plt.title('Shifting the distribution of the minimum measurable effect gives {0}% chance of a significant result'.format(np.round(power*100, 3)), fontsize=18)

    plt.axvline(critical_point_ri, color='red', linestyle='--', linewidth=6)

    # We can probably actually make these label locations super precise. These are heuristics which will only work in this case...
    area_labels = ['  Effect distribution \nunder null hypothesis', '{0}%'.format(np.round(significance_level, 2)), '{0}%'.format(np.round(power*100, 3))]
    y_vals = ax.get_yticks()
    y_diff = y_vals[1] - y_vals[0]
    y_loc = y_diff / 2.0
    ax.text(critical_point_ri + 0.2*critical_point_ri, y_loc-3, area_labels[1], fontsize=18)
    ax.text(critical_point_ri - 1.7*critical_point_ri, y_loc+5, area_labels[0], fontsize=18)
    ax.text(critical_point_ri + 0.5*critical_point_ri, y_loc+10, area_labels[2], fontsize=18)
    plt.text(critical_point_ri + 0.05*critical_point_ri,
             np.median(ax.get_yticks()),
             "Critical value: {0}".format(np.round(critical_point_ri, 4)),
             rotation=90,
             verticalalignment='center',
             fontsize=16)

    kde_2_x, kde_2_y = ax.lines[1].get_data()
    ax.fill_between(kde_2_x, kde_2_y, where=(kde_2_x >= critical_point_ri), interpolate=True, color='blue', alpha=0.3)
    kde_x, kde_y = ax.lines[0].get_data()
    ax.fill_between(kde_x, kde_y, where=(kde_x >= critical_point_ri), interpolate=True, color='red', alpha=0.5)

    # TODO: Make the plot locations less heuristic.
    left_x_loc = kde_x[np.argmax(kde_y)]
    # right_x_loc = kde_2_x[np.argmax(kde_2_y)]
    right_x_loc = left_x_loc + mme
    y_loc_mme = np.max(kde_y)
    plt.arrow(x=left_x_loc, y=y_loc_mme, dx=right_x_loc, dy=0, head_width=2, head_length=0.001, linewidth=2, color='blue', length_includes_head=False)
    plt.text(x=(left_x_loc+right_x_loc)/9, y=y_loc_mme - 0.05*y_loc_mme, s="Effect Distribution \nShifted by MME: {0}".format(np.round(mme, 3)), fontsize=16)

    plt.tight_layout()
    plt.savefig(file_name)


def run(historical_data_conversion_column: str = None, make_visualizations: bool = False, number_of_data_points: int = 1000, null_conversion_rate: float = 0.01,
        path_to_data: str = None, significance_level: float = 0.05, power_level: float = 0.8, number_of_simulations: int = 10000, run_in_parallel: bool = True,
        random_seed: int = None, make_mock_data: bool = False, output_path: str = None) -> bool:

    # Generate the current time for file saving convention
    current_time = strftime("%Y-%m-%d_%H%M%S", gmtime())

    # Check the output directory. If it's not None and doesn't exist, create it
    if make_visualizations:
        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)

    # Verify that path_to_data is None and mock_data is True
    if path_to_data is None:
        assert make_mock_data is True, "If no path to historical data is provided, you must set make_mock_data to True to generate a toy dataset"
    else:
        assert make_mock_data is False, "A path to historical data has been provided. In this case, make_mock_data must be False"

    if path_to_data is not None:
        df = load_data(path_to_data=path_to_data)

    if make_mock_data is True:
        df = make_mock_historical_data(number_of_data_points=number_of_data_points, null_conversion_rate=null_conversion_rate)
        historical_data_conversion_column = 'data_unit_converted'

    # Run the effect size simulation
    print("Running randomization inference for {0} simulations".format(number_of_simulations))
    simulated_effect_sizes_ri = randomization_inference(df=df,
                                                        number_of_simulations=number_of_simulations,
                                                        data_unit_col_name=historical_data_conversion_column,
                                                        run_in_parallel=run_in_parallel,
                                                        random_seed=random_seed)
    print("Completed simulations. Generated {0} data points".format(len(simulated_effect_sizes_ri)))

    # Calculate the critical value. This comes from the desired level of significance
    q_significance = 1 - significance_level
    critical_point_ri = np.quantile(simulated_effect_sizes_ri, q_significance)

    # Compute the mme:
    mme_ = compute_mme(critical_value=critical_point_ri, simulated_effects_under_null_distribution=simulated_effect_sizes_ri, power=power_level)

    # Plot the distributions if needed
    if make_visualizations:
        minimum_measurable_effect_ri = shift_null_distribution_by_mme(simulated_effects_under_null_distribution=simulated_effect_sizes_ri, mme=mme_)

        plot_simulated_null_distribution(simulated_effect_sizes_ri=simulated_effect_sizes_ri,
                                         critical_value=critical_point_ri,
                                         current_time=current_time,
                                         p_value=None,
                                         height=10,
                                         aspect=1.5,
                                         significance_level=significance_level,
                                         save_directory=output_path)

        visualize_mme(simulated_effect_sizes_ri=simulated_effect_sizes_ri,
                      minimum_measurable_effect_ri=minimum_measurable_effect_ri,
                      power=power_level,
                      critical_point_ri=critical_point_ri,
                      mme=mme_,
                      significance_level=significance_level,
                      current_time=current_time,
                      save_directory=output_path)

    print('The minimum measurable effect size is given a desired power of {0}% and a significance level of {1}%: {2}'.format(np.round(power_level*100, 2),
                                                                                                                             np.round(q_significance*100, 2),
                                                                                                                             np.round(mme_, 3)))

    return True


if __name__ == '__main__':
    conversion_column_name = 'made_valid_decision'
    plot_distributions = True
    number_of_data_points_to_generate = 50000
    baseline_conversion_rate = 0.016
    historical_data_path = '/Users/danielokeeffe/src/github.com/Shopify/misc/end_of_trial_exp/data/all_trials_auto_installed_july_2022.csv'
    sig_level = 0.05
    power = 0.8
    number_of_simulations_to_run = 10000
    simulate_in_parallel = False
    seed = None
    make_toy_data_set = False
    save_path = '../../data/figs/end_of_trial_exp_20220812b/'

    result = run(historical_data_conversion_column=conversion_column_name,
                 make_visualizations=plot_distributions,
                 number_of_data_points=number_of_data_points_to_generate,
                 null_conversion_rate=baseline_conversion_rate,
                 path_to_data=historical_data_path,
                 significance_level=sig_level,
                 power_level=power,
                 number_of_simulations=number_of_simulations_to_run,
                 run_in_parallel=simulate_in_parallel,
                 random_seed=seed,
                 make_mock_data=make_toy_data_set,
                 output_path=save_path)

    if result:
        sys.exit()
