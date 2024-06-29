"""

 _____      _                             _           _   _____ _                  _____           _             _   _ _   _ _
|_   _|    | |                           | |         | | |_   _(_)                /  ___|         (_)           | | | | | (_) |
  | | _ __ | |_ ___ _ __ _ __ _   _ _ __ | |_ ___  __| |   | |  _ _ __ ___   ___  \ `--.  ___ _ __ _  ___  ___  | | | | |_ _| |___
  | || '_ \| __/ _ \ '__| '__| | | | '_ \| __/ _ \/ _` |   | | | | '_ ` _ \ / _ \  `--. \/ _ \ '__| |/ _ \/ __| | | | | __| | / __|
 _| || | | | ||  __/ |  | |  | |_| | |_) | ||  __/ (_| |   | | | | | | | | |  __/ /\__/ /  __/ |  | |  __/\__ \ | |_| | |_| | \__ \
 \___/_| |_|\__\___|_|  |_|   \__,_| .__/ \__\___|\__,_|   \_/ |_|_| |_| |_|\___| \____/ \___|_|  |_|\___||___/  \___/ \__|_|_|___/
                                   | |
                                   |_|

A collection of utility functions to automate quasi-experiments (i.e. non-randomized data) using interrupted time series

Author: Dan (okeeffed090@gmail.com)

V1.0.0
"""

# Import packages
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.ticker as mtick
import statsmodels.formula.api as smf
import causalimpact
import random
import datetime
import statsmodels as sms
import statsmodels.api as sm

from matplotlib import style
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import pmdarima as pm
from pmdarima import model_selection
from time import gmtime, strftime
from typing import Union
from tqdm import tqdm

# set the plot style
style.use('fivethirtyeight')


class InterruptedTimeSeries:
    """
    Fill this in later
    """

    def __init__(self, treatment_date, date_col):
        """

        :param treatment_date: The date on which the treatment occurred. This is expected to be a python datetime object
        :param date_col: The name of the column containing the dates
        """
        self.supported_model_types = ('interrupted_time_series', 'naive_arima_interrupted_time_series', 'auto_arima_interrupted_time_series', 'regression_discontinuity')
        self.interrupted_time_series_vars = ('outcome', 'T', 'D', 'P')
        self.treatment_date = treatment_date
        self.date_col = date_col

    def interrupted_time_series(self, df, model_type):
        # TODO: this will be a general function to run the entire analysis. It should take a model type, data, and column definition + plot preferences (and saving preferences) and run everything

        if outcome_name is None:
            outcome_name = 'outcome'

        if treatment_name is None:
            treatment_name = 'treatment'

        if save_path is None:
            save_path = os.getcwd()

        # Generate the current time for plot labeling

        # Aggregate data is instructed to

        # Prep data

        # Plot the treatment pre-post periods

        # Build the model object

        # Return and interpret results. This will need to be a separate flow for different model types, probably

        # Make recommendations for a better approach

        # If it looks like we need to, auto-produce autocorrelation plots

        # Compute counterfactuals

        # Plot and return counterfactuals along with effect size estimates

        pass

    def prep_data_for_interrupted_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to prepare data for the interrupted time series analysis.  Specifically, we need to add:

        1. T: a continuous variable which indicates the time passed from the start of the observational period
        2. D: a dummy variable indicated observations collected before (D = 0) or after (D = 1) the intervention
        3. P: a variable indicating the time passed since the intervention has occurred (before the intervention has occurred, P = 0)

        :param df: DataFrame with the time series data you want to study. At a minimum requires a date column which is pre-aggregated to the grain you want to understand (e.g. one
                   row per day) and another column with the observed quantity

        :return: A DataFrame with the necessary derived columns for an interrupted time series model
        """

        # Make a copy of the input DataFrame to not inadvertently modify the original in-place
        df_ = df.copy()

        # ensure that all supplied dates are unique. i.e. there are no rows with duplicated dates
        assert not max(df_.duplicated(subset=self.date_col)), "Each row must be a unique date. Please pre-aggregate your data appropriately"

        # We need to ensure that the DataFrame is sorted by the date_col
        df_ = df_.sort_values(by=self.date_col, ascending=True)

        # Add the post treatment variable. This is D
        df_['D'] = df_[self.date_col].apply(lambda x: 0 if x < self.treatment_date else 1)

        # Add the running variable. This is P
        df_['P'] = df_[self.date_col] - self.treatment_date
        df_['P'] = df_['P'].apply(lambda x: x.days)
        df_['P'] = df_['P'].apply(lambda x: 0 if x < 0 else x)

        # Add the observation counter. This is T. It's a variable that counts from 1 to N for each N observations (one per time period)
        df_ = df_.reset_index()
        df_ = df_.rename(index=str, columns={'index': 'T'})
        df_['T'] = df_['T'].astype(int)
        df_['T'] = df_['T'] + 1

        return df_

    def plot_pre_post_treatment_periods(self, df: pd.DataFrame, current_time: str, outcome_col: str, outcome_name: str, treatment_name: str, save_path: str):
        """
        Function to plot the pre-post period.  No model is applied here, this is just for visual inspection

        :param df: DataFrame with the data you want to investigate. Should be the output from prep_data_for_interrupted_time_series as some column names are assumed
        :param current_time: String indicating the time the analysis was started. This is used to name output files.
        :param outcome_col: Name of the column that has the outcome variable
        :param outcome_name: Name of the outcome.
        :param treatment_name: Name of the treatment.
        :param save_path: Optional path to where you want to save the resulting figure. If none, the plot will be saved in the current working directory
        :return:
        """
        plot_title = "{0} pre/post {1}".format(outcome_name, treatment_name)

        ax = df.query("D==0").plot.scatter(x=self.date_col, y=outcome_col, figsize=(12, 8), label='Pre-treatment', facecolors='none', edgecolors='steelblue', linewidths=2)
        df.query("D==1").plot.scatter(x=self.date_col, y=outcome_col, label='Post-Treatment', ax=ax, facecolors='none', edgecolors='green', linewidths=2)
        ax.axvline(x=self.treatment_date, linestyle='--', ymax=1, color='red', label='Intervention started')
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel(outcome_name, fontsize=16)
        plt.xticks(rotation=45)
        ax.legend()

        file_name = 'pre_post_{0}_for_metric_{1}_{2}.png'.format(treatment_name, outcome_name, current_time)
        file_name = os.path.join(save_path, file_name)

        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    def generate_model(self, df: pd.DataFrame, variable_name_dict: dict, arima_params_dict: dict, model_type: str) -> Union[sm.regression.linear_model.RegressionResultsWrapper,
                                                                                                                            sms.tsa.arima.model.ARIMAResultsWrapper,
                                                                                                                            sms.tsa.statespace.sarimax.SARIMAXResultsWrapper]:
        """
        Function to generate the desired model to use for effect size and counterfactual estimation. This will set the model specifications for the desired type and fit the model
        to the supplied data in df. It is strongly recommended that you pre-process your data via the prep_data_for_interrupted_time_series method first.

        :param df: DataFrame with the data you want to investigate. Should be the output from prep_data_for_interrupted_time_series as some column names are assumed
        :param variable_name_dict: Name of the variables in your DataFrame as keys and how each one maps to one of outcome, T, D, P as values.  See the
                                   prep_data_for_interrupted_time_series method for definitions of these variables. It is recommended to just pre-process your data through
                                   prep_data_for_interrupted_time_series. In this all you need to do is specify the outcome column name.
                                   i.e. variable_name_dict = {'outcome': OUTCOME_COLUMN_NAME_IN_DATA, 'T': 'T', 'D': 'D', 'P': 'P'}
        :param arima_params_dict: Dictionary with parameters to pass to ARIMA models.  naive_arima_interrupted_time_series will look for two keys: order and seasonal_order. Both
                                  are expected to have tuples as items.  See here for a definition of what goes into the tuples:
                                  https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
                                  auto_arima_interrupted_time_series will look for two other keys (and ignore order and seasonal_order); seasonal_auto_arima and
                                  seasonal_period_auto_arima. seasonal_auto_arima is a Boolean. If True, the model will auto-fit a SARIMA model, otherwise will default to ARIMA.
                                  seasonal_period_auto_arima is the periodicity of your data (e.g. 7 days).  Setting this to 1 will override seasonal_auto_arima and only an ARIMA
                                  model will be fit
        :param model_type: Type of model you want to fit.  See then __init__ method for a list of currently supported model types

        :return: A fit model object
        """

        assert model_type in self.supported_model_types, "model type {0} not one of the supported types: {1}".format(model_type, self.supported_model_types)
        # Make sure the input dict has the necessary variables
        assert all(name in self.supported_model_types for name in
                   variable_name_dict), "input variables doesn't contain all necessary variable for interrupted time series. Required variables: {0}".format(self.supported_model_types)

        if model_type == 'interrupted_time_series':

            model_vars = [variable_name_dict[i] for i in self.supported_model_types]
            model_str = "{0} ~ {1} + {2} + {3}".format(*model_vars)

            model_obj = smf.ols(formula=model_str, data=df)

            model_ = model_obj.fit()

        if model_type == 'naive_arima_interrupted_time_series':

            outcome_col = variable_name_dict['outcome']
            covariate_cols = [val for key, val in variable_name_dict.items() if key != 'outcome']

            try:
                arima_order = arima_params_dict['order']
            except KeyError:
                print("WARNING: No ARIMA order parameters found. Default is (0, 0, 0)")
                arima_order = (0, 0, 0)
            try:
                arima_seasonal_order = arima_params_dict['seasonal_order']
            except KeyError:
                print("INFO: Fitting non-seasonal ARIMA")
                arima_seasonal_order = (0, 0, 0, 0)

            model_obj = ARIMA(df[outcome_col], df[covariate_cols], order=arima_order, seasonal_order=arima_seasonal_order)

            model_ = model_obj.fit()

        if model_type == 'auto_arima_interrupted_time_series':

            outcome_col = variable_name_dict['outcome']
            covariate_cols = [val for key, val in variable_name_dict.items() if key != 'outcome']

            try:
                seasonal_ = arima_params_dict['seasonal_auto_arima']
            except KeyError:
                print('INFO: Fitting non-seasonal auto ARIMA')
                seasonal_ = False
            try:
                seasonal_period = arima_params_dict['seasonal_period_auto_arima']
            except KeyError:
                print("INFO: Setting default seasonal period to 1 (i.e. non-seasonal)")
                seasonal_period = 1

            model_obj = pm.auto_arima(y=df[outcome_col], X=df[covariate_cols], seasonal=seasonal_, m=seasonal_period, supress_warnings=True, trace=True)

            model_ = model_obj.arima_res_

        return model_

    def ols_model_interpreter(self, df: pd.DataFrame, variable_name_dict, model_: Union[sm.regression.linear_model.RegressionResultsWrapper, sms.tsa.arima.model.ARIMAResultsWrapper, sms.tsa.statespace.sarimax.SARIMAXResultsWrapper], model_type: str, save_path: str, current_time: str, alpha: float = 0.05):
        """

        :param df:
        :param variable_name_dict:
        :param model_:
        :param model_type:
        :param save_path:
        :param alpha:
        :param current_time:

        :return:
        """

        assert model_type in self.supported_model_types, "model type {0} not one of the supported types: {1}".format(model_type, self.supported_model_types)
        assert all(name in self.supported_model_types for name in
                   variable_name_dict), "input variables doesn't contain all necessary variable for interrupted time series. Required variables: {0}".format(self.supported_model_types)

        post_treatment_col = variable_name_dict['D']
        outcome_col = variable_name_dict['outcome']

        print(model_.summary())

        # get the effect size
        effect_size_abs = model_.params['D']
        effect_size_abs_lower = model_.conf_int(alpha=alpha, cols=None)[0][post_treatment_col]
        effect_size_abs_upper = model_.conf_int(alpha=alpha, cols=None)[1][post_treatment_col]

        effect_size_rel = (effect_size_abs / df.query("{0}==0".format(post_treatment_col))[outcome_col].mean())
        effect_size_rel_lower = (effect_size_abs_lower / df.query("{0}==0".format(post_treatment_col))[outcome_col].mean())
        effect_size_rel_upper = (effect_size_abs_upper / df.query("{0}==0".format(post_treatment_col))[outcome_col].mean())

        # Get p-value
        p_val = model_.pvalues[post_treatment_col]
        is_stat_sig = p_val < alpha

        print('\n\n')

        effect_abs_str = "Absolute effect size estimated at: {0} {1}% CI: ({2} - {3})".format(np.round(effect_size_abs, 2),
            np.round(int((1 - alpha) * 100), 2),
            np.round(effect_size_abs_lower, 2),
            np.round(effect_size_abs_upper, 2))
        effect_rel_str = "Absolute effect size estimated at: {0}% {1}% CI: ({2}% - {3}%)".format(np.round(effect_size_rel * 100, 2),
            np.round(int((1 - alpha) * 100), 2),
            np.round(effect_size_rel_lower * 100, 2),
            np.round(effect_size_rel_upper * 100, 2))
        if is_stat_sig:
            stat_sig_str = "P-value: {0}. The effect is statistically significant".format(np.round(p_val, 3))
        else:
            stat_sig_str = "P-value: {0}. No evidence for significant effect".format(np.round(p_val, 3))

        print(effect_abs_str)
        print('\n')
        print(effect_rel_str)
        print('\n')
        print(stat_sig_str)
        # TODO: consolidate this. We can actually combine almost everything for all model types at this point
        if model_type == 'interrupted_time_series':

            # Get the Durban-Watson statistic
            dw_stat = durbin_watson(model_.resid)

            # Get the Jarque Bera statistics
            jb_prob = jarque_bera(model_.resid)[1]

            # Verify assumptions of OLS
            if jb_prob < 0.05:
                residual_normality_verified = False
            else:
                residual_normality_verified = True

            if dw_stat < 1.5:
                residual_independence_verified = False
                residual_independence_violation = 'positive auto-correlation'
            if dw_stat > 2.5:
                residual_independence_verified = False
                residual_independence_violation = 'negative auto-correlation'
            else:
                residual_independence_verified = True
                residual_independence_violation = None

            if not residual_normality_verified:
                print("WARNING: Jarque-Bera statistic = {0}".format(np.round(jb_stat, 3)))
                print('Evidence that residuals do not follow a normal distribution. Revise the choice of a least squares model')
            if not residual_independence_verified:
                print("WARNING: Durban-Watson statistic = {0}".format(np.round(dw_stat, 3)))
                print("evidence for {0} in residuals.  Residuals may not be independent. Revise the choice of a least square model".format(residual_independence_violation))
                print("Generating auto-correlation and partial auto-correlation plots for review...")

                file_name_auto_corr = 'autocorrelation_of_residuals_{0}.png'.format(current_time)
                file_name_auto_corr = os.path.join(save_path, file_name_auto_corr)
                sm.graphics.tsa.plot_acf(model_.resid, lags=10)
                plt.savefig(file_name_auto_corr)

                file_name_partial_auto_corr = 'partial_autocorrelation_of_residuals_{0}.png'.format(current_time)
                file_name_partial_auto_corr = os.path.join(save_path, file_name_partial_auto_corr)
                sm.graphics.tsa.plot_pacf(model_.resid, lags=10)
                plt.savefig(file_name_partial_auto_corr)

        if model_type in ('naive_arima_interrupted_time_series', 'auto_arima_interrupted_time_series'):

            # Get the Jarque-Bera statistics
            jb_prob = jarque_bera(model_.resid)[1]

            # Get the Ljung-Box Q statistic
            q_stat_prob = pd.read_html(model_.summary().tables[2].as_html(), index_col=0)[0][1]['Prob(Q):']

            # Verify assumptions of OLS
            if jb_prob < 0.05:
                residual_normality_verified = False
            else:
                residual_normality_verified = True

            if q_stat_prob < 0.05:
                residual_independence_verified = False
            else:
                residual_independence_verified = True

            if not residual_normality_verified:
                print("WARNING: Jarque-Bera statistic = {0}".format(np.round(jb_stat, 3)))
                print('Evidence that residuals do not follow a normal distribution. Revise the choice of a least squares model')










