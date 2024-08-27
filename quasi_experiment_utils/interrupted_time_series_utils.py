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
import warnings

from matplotlib import style
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import pmdarima as pm
from pmdarima import model_selection
from time import gmtime, strftime
from typing import Union
from tqdm import tqdm

warnings.filterwarnings('ignore')

# set the plot style
style.use('fivethirtyeight')


class InterruptedTimeSeries:
    """
    Class to instantiate an InterruptedTimeSeries object. This is the basic object for running and storing results from a variety of interrupted time series methods

    This class exposes several options for modeling your data. Currently, the supported models are:

        1. interrupted_time_series:
            a. This fits a simple Ordinary Least Squares (OLS) model. It regresses the observed outcome on three variables:
                i. T: a variable which indicates the time passed from start of the observational period
                ii. D: a dummy indicator variable identifying observations collected before (D = 0) or after (D = 1) the intervention
                iii. P: a variable indicating the time passes since the intervention has occurred (before the intervention has occurred, P is equal to 0)
            b. The fit model is then outcome ~ T + D + P
            c. The model coefficient for D is then the estimated impact of the intervention
        2. naive_arima_interrupted_time_series:
            a. Replaces the simple OLS model with a simple ARIMA model
            b. This can be useful if there is considerable auto-correlation in your data. This can cause the residuals of an OLS model to not be independent, meaning that a model
               assumption is violated
            c. The order and seasonal order parameters must be supplied to this model
        3. auto_arima_interrupted_time_series:
            a. This is the same as the naive_arima_interrupted_time_series except the order and seasonal order parameters will be selected for you
        4. More will be added in the future.

        Given the input data, required parameters, and model type, the method interrupted_time_series in this class will:

        1. Provide a plot of the pre-/post-intervention period for visual inspection
        2. Generate the desired model
        3. Provide a summary of the estimated effect including a plot of the estimate counterfactuals vs observations
        4. Validate the fit model by checking a small number of model assumptions
        5. Provide recommendations of what to try next if model assumptions are violated

        You can, loosely, think of steps 2, 3, and 4 above as a simplified version of a causal inference work flow: indentify, estimate, refute.

        Each step in the process is handled by a specific function. See the documentation for these functions for specific details about how all this works.

    """

    def __init__(self, treatment_date):
        """

        :param treatment_date: The date on which the treatment occurred. This is expected to be a python datetime object
        """
        self.supported_model_types = ('interrupted_time_series',
                                      'naive_arima_interrupted_time_series',
                                      'auto_arima_interrupted_time_series')
        # self.supported_model_types = ('interrupted_time_series',
        #                               'naive_arima_interrupted_time_series',
        #                               'auto_arima_interrupted_time_series',
        #                               'regression_discontinuity',
        #                               'fuzzy_regression_discontinuity')
        self.interrupted_time_series_vars = ('outcome', 'T', 'D', 'P')
        # self.interrupted_time_series_vars = ('T', 'D', 'P')
        self.treatment_date = treatment_date
        self.date_col = None
        self.treatment_name = None
        self.outcome_name = None
        self.effect_size_abs = None
        self.effect_size_abs_lower = None
        self.effect_size_abs_upper = None
        self.effect_size_rel = None
        self.effect_size_rel_lower = None
        self.effect_size_rel_upper = None
        self.p_value = None
        self.counterfactual = None
        self.outcome_predictions = None
        # Initialize the order and seasonal order parameters for an auto arima model
        # If auto-arima is used, these will be updated to match the best order and
        # seasonal order parameter sets identified.
        self.auto_arima_fit_order = (0, 0, 0)
        self.auto_arima_fit_seasonal_order = (0, 0, 0, 0)

    def interrupted_time_series(self, df: pd.DataFrame, model_type: str, date_col: str, outcome_col: str, outcome_name: str = None, treatment_name: str = None, arima_params_dict: dict = None, aggregate_data: bool = False, save_path: str = None, figsize: tuple = (16, 10), alpha: float = 0.05):
        """
        Function to handle an interrupted time series analysis from end to end.  All that is required is an input dataframe with a datetime like column, and an outcome column.
        Ideally, the data represents one aggregate observation per time period (e.g. total sales per day).  If this is not the case, there is an option to auto-aggregate the data
        for you.  However, this currently only work with summation.

        This method exposes several options for modeling your data. See the InterruptedTimeSeries class docstrings for specific details.

        Given the input data, required parameters, and model type, this method will:

        1. Provide a plot of the pre-/post-intervention period for visual inspection
        2. Generate the desired model
        3. Provide a summary of the estimated effect including a plot of the estimate counterfactuals vs observations
        4. Validate the fit model by checking a small number of model assumptions
        5. Provide recommendations of what to try next if model assumptions are violated

        You can, loosely, think of steps 2, 3, and 4 above as a simplified version of a causal inference work flow: indentify, estimate, refute.

        :param df: DataFrame with the time series data you want to study. At a minimum requires a date column and another column with the observed quantity
        :param model_type: String indicating the model type you want for the counterfactual.  See the supported_model_types variable in the __init__ method to see which model types
                           are currently supported
        :param date_col: The name of the column containing the dates
        :param outcome_name: Name of the outcome
        :param outcome_col: Name of the column in the input DataFrame that has the outcome observations
        :param treatment_name: Name of the treatment
        :param arima_params_dict: Dictionary with parameters to pass to ARIMA models.  naive_arima_interrupted_time_series will look for two keys: order and seasonal_order. Both
                                  are expected to have tuples as items.  See here for a definition of what goes into the tuples:
                                  https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
                                  auto_arima_interrupted_time_series will look for two other keys (and ignore order and seasonal_order); seasonal_auto_arima and
                                  seasonal_period_auto_arima. seasonal_auto_arima is a Boolean. If True, the model will auto-fit a SARIMA model, otherwise will default to ARIMA.
                                  seasonal_period_auto_arima is the periodicity of your data (e.g. 7 days).  Setting this to 1 will override seasonal_auto_arima and only an ARIMA
                                  model will be fit.  If this is None, then all order and seasonal order parameters will be assumed 0.
        :param aggregate_data: Boolean. If your data is not already pre-aggregated by datetime-grain/outcome, you can set this to True and have the aggregation done for you.  This
                               currently only supports summation as an aggregation function
        :param save_path: Optional path to where you want to save the resulting figure. If none, the plot will be saved in the current working directory
        :param figsize: Optional tuple containing the size of the output figure which plots model predictions and counterfactuals. Default is (16, 10).
        :param alpha: The probability of rejecting the null hypothesis when the null hypothesis is true. i.e. 1 - alpha = significance leve

        :return: Nothing
        """

        self.date_col = date_col

        if outcome_name is None:
            self.outcome_name = 'outcome'
        else:
            self.outcome_name = outcome_name

        if treatment_name is None:
            self.treatment_name = 'treatment'
        else:
            self.treatment_name = treatment_name

        if save_path is None:
            save_path = os.getcwd()

        if arima_params_dict is None:
            arima_params_dict = {}

        assert type(arima_params_dict) == dict, "the ARIMA parameters must be input as a dictionary. e.g. {'order': (1, 1, 1), 'seasonal_order': (1, 2, 3, 4)}"

        # Make a copy of the input DataFrame to not inadvertently modify the original in-place
        df_ = df.copy()

        # Generate the current time for plot labeling
        current_time = strftime('%Y-%m-%d_%H%M%S', gmtime())

        # Aggregate data if instructed to
        if aggregate_data:
            print("WARNING: Aggregating time series data. Only the outcome column {0} and the date {1} column will be retained".format(outcome_col, self.date_col))
            print("WARNING: Aggregation assumes that the sum of the outcome column {0} per timestep is required".format(outcome_col))
            df_ = df_[[outcome_col, self.date_col]].groupby(self.date_col, as_index=False).sum()

        # Prep data
        df_ = self.prep_data_for_interrupted_time_series(df=df_)
        variable_name_dict = {'outcome': outcome_col, 'T': 'T', 'D': 'D', 'P': 'P'}

        # Plot the treatment pre-post periods
        self.plot_pre_post_treatment_periods(df=df_, variable_name_dict=variable_name_dict, current_time=current_time, save_path=save_path)

        # Build the model object
        model_ = self.generate_model(df=df_, variable_name_dict=variable_name_dict, arima_params_dict=arima_params_dict, model_type=model_type, refit_=False)

        # Return and interpret results.
        violated_ = self.ols_model_interpreter(df=df_,
                                               variable_name_dict=variable_name_dict,
                                               model_=model_,
                                               model_type=model_type,
                                               save_path=save_path,
                                               current_time=current_time,
                                               alpha=alpha)

        if violated_:
            print("WARNING: Some or all model assumptions tested have been invalidated!")
            print('\n')

        # Compute counterfactuals and plot along with effect size estimates
        self.counterfactual, self.outcome_predictions = self.calculate_counterfactuals(df=df_,
                                                                                       arima_params_dict=arima_params_dict,
                                                                                       variable_name_dict=variable_name_dict,
                                                                                       model_=model_,
                                                                                       model_type=model_type,
                                                                                       save_path=save_path,
                                                                                       figsize=figsize,
                                                                                       alpha=alpha,
                                                                                       current_time=current_time)

        print("Process complete")

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

    def plot_pre_post_treatment_periods(self, df: pd.DataFrame, variable_name_dict: dict, current_time: str, save_path: str):
        """
        Function to plot the pre-post period.  No model is applied here, this is just for visual inspection

        :param df: DataFrame with the data you want to investigate. Should be the output from prep_data_for_interrupted_time_series as some column names are assumed
        :param current_time: String indicating the time the analysis was started. This is used to name output files.
        :param variable_name_dict: Name of the variables in your DataFrame as keys and how each one maps to one of outcome, T, D, P as values.  See the
                                   prep_data_for_interrupted_time_series method for definitions of these variables. It is recommended to just pre-process your data through
                                   prep_data_for_interrupted_time_series. In this all you need to do is specify the outcome column name.
                                   i.e. variable_name_dict = {'outcome': OUTCOME_COLUMN_NAME_IN_DATA, 'T': 'T', 'D': 'D', 'P': 'P'}
        :param save_path: Optional path to where you want to save the resulting figure. If none, the plot will be saved in the current working directory

        :return: Nothing
        """
        plot_title = "{0} pre/post {1}".format(self.outcome_name, self.treatment_name)

        outcome_col = variable_name_dict['outcome']
        post_treatment_col = variable_name_dict['D']

        ax = df.query("{0}==0".format(post_treatment_col)).plot.scatter(x=self.date_col, y=outcome_col, figsize=(12, 8), label='Pre-treatment', facecolors='none', edgecolors='steelblue', linewidths=2)
        df.query("{0}==1".format(post_treatment_col)).plot.scatter(x=self.date_col, y=outcome_col, label='Post-Treatment', ax=ax, facecolors='none', edgecolors='green', linewidths=2)
        ax.axvline(x=self.treatment_date, linestyle='--', ymax=1, color='red', label='Intervention started')
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel(self.outcome_name, fontsize=16)
        plt.xticks(rotation=45)
        ax.legend()

        file_name = 'pre_post_{0}_for_metric_{1}_{2}.png'.format(self.treatment_name, self.outcome_name, current_time)
        file_name = os.path.join(save_path, file_name)

        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    def generate_model(self, df: pd.DataFrame, variable_name_dict: dict, arima_params_dict: dict, model_type: str, refit_: bool = False) -> Union[sm.regression.linear_model.RegressionResultsWrapper,
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
        :param refit_: Boolean. True if a call to this function is intend to refit and previously fit model.  Otherwise, false.

        :return: A fit model object
        """

        # TODO: Move these checks to the interrupted_time_series method and only do they once? Might want to leave them here in case someone tries to use this as a standalone?
        assert model_type in self.supported_model_types, "model type {0} not one of the supported types: {1}".format(model_type, self.supported_model_types)
        # Make sure the input dict has the necessary variables
        # This only applies if this model is not meant to be a refit for estimating a counterfactual
        # In that case, both D and P would not exist.
        if not refit_:
            assert all(name in self.interrupted_time_series_vars for name in
                       variable_name_dict), "input variables doesn't contain all necessary variable for interrupted time series. Required variables: {0}".format(self.interrupted_time_series_vars)

        if model_type == 'interrupted_time_series':

            model_vars = [variable_name_dict[i] for i in self.interrupted_time_series_vars]
            model_str = "{0} ~ {1} + {2} + {3}".format(*model_vars)

            model_obj = smf.ols(formula=model_str, data=df)

            model_ = model_obj.fit()

        if model_type == 'naive_arima_interrupted_time_series':

            outcome_col = variable_name_dict['outcome']
            if not refit_:
                covariate_cols = [val for key, val in variable_name_dict.items() if key != 'outcome']
            else:
                covariate_cols = variable_name_dict['T']

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

            # We also need to get the oder and seasonal order parameters out of this thing
            if not refit_:
                self.auto_arima_fit_order = model_obj.order
                self.auto_arima_fit_seasonal_order = model_obj.seasonal_order

        return model_

    def ols_model_interpreter(self, df: pd.DataFrame, variable_name_dict, model_: Union[sm.regression.linear_model.RegressionResultsWrapper, sms.tsa.arima.model.ARIMAResultsWrapper, sms.tsa.statespace.sarimax.SARIMAXResultsWrapper], model_type: str, save_path: str, current_time: str, alpha: float = 0.05) -> bool:
        """
        Function to help interpret the results of an interrupted time series analysis.  Works with all currently supported models.  Will test a minimum number of the assumptions
        of the underlying regression models.  This should be thought of as a refutation step in a causal inference workflow, although it's not quite that robust at this point.
        The function will also return recommendations for what to do next should any of the assumptions be violated in your model.

        :param df: DataFrame with the data you want to investigate. Should be the output from prep_data_for_interrupted_time_series as some column names are assumed
        :param variable_name_dict: Name of the variables in your DataFrame as keys and how each one maps to one of outcome, T, D, P as values.  See the
                                   prep_data_for_interrupted_time_series method for definitions of these variables. It is recommended to just pre-process your data through
                                   prep_data_for_interrupted_time_series. In this all you need to do is specify the outcome column name.
                                   i.e. variable_name_dict = {'outcome': OUTCOME_COLUMN_NAME_IN_DATA, 'T': 'T', 'D': 'D', 'P': 'P'}
        :param model_: A fit model object. Should be the output of generate_model
        :param model_type: Type of model you want to fit.  See then __init__ method for a list of currently supported model types
        :param save_path: Optional path to where you want to save any generated figures. If none, the plots will be saved in the current working directory
        :param alpha: The probability of rejecting the null hypothesis when the null hypothesis is true. i.e. 1 - alpha = significance level
        :param current_time: String containing the current datetime. This is used for generating plot filenames.

        :return: Boolean. True if any of the tested model assumptions have been invalidated. Otherwise, False.
        """
        # TODO: I can remove this here. This is done above and no one should be using this method as a standalone
        assert model_type in self.supported_model_types, "model type {0} not one of the supported types: {1}".format(model_type, self.supported_model_types)
        assert all(name in self.interrupted_time_series_vars for name in
                   variable_name_dict), "input variables doesn't contain all necessary variable for interrupted time series. Required variables: {0}".format(self.interrupted_time_series_vars)

        post_treatment_col = variable_name_dict['D']
        outcome_col = variable_name_dict['outcome']

        print(model_.summary())

        # get the effect size
        self.effect_size_abs = model_.params['D']
        self.effect_size_abs_lower = model_.conf_int(alpha=alpha, cols=None)[0][post_treatment_col]
        self.effect_size_abs_upper = model_.conf_int(alpha=alpha, cols=None)[1][post_treatment_col]

        self.effect_size_rel = (self.effect_size_abs / df.query("{0}==0".format(post_treatment_col))[outcome_col].mean())
        self.effect_size_rel_lower = (self.effect_size_abs_lower / df.query("{0}==0".format(post_treatment_col))[outcome_col].mean())
        self.effect_size_rel_upper = (self.effect_size_abs_upper / df.query("{0}==0".format(post_treatment_col))[outcome_col].mean())

        # Get p-value
        self.p_value = model_.pvalues[post_treatment_col]
        is_stat_sig = self.p_value < alpha

        print('\n\n')

        effect_abs_str = "Absolute effect size estimated at: {0} {1}% CI: ({2} - {3})".format(
                                                                                              np.round(self.effect_size_abs, 2),
                                                                                              np.round(int((1 - alpha) * 100), 2),
                                                                                              np.round(self.effect_size_abs_lower, 2),
                                                                                              np.round(self.effect_size_abs_upper, 2)
                                                                                              )
        effect_rel_str = "Relative effect size estimated at: {0}% {1}% CI: ({2}% - {3}%)".format(
                                                                                                 np.round(self.effect_size_rel * 100, 2),
                                                                                                 np.round(int((1 - alpha) * 100), 2),
                                                                                                 np.round(self.effect_size_rel_lower * 100, 2),
                                                                                                 np.round(self.effect_size_rel_upper * 100, 2)
                                                                                                )
        if is_stat_sig:
            stat_sig_str = "P-value: {0}. The effect is statistically significant".format(np.round(self.p_value, 3))
        else:
            stat_sig_str = "P-value: {0}. No evidence for significant effect".format(np.round(self.p_value, 3))

        print(effect_abs_str)
        print('\n')
        print(effect_rel_str)
        print('\n')
        print(stat_sig_str)
        print('\n')

        # Get the Durban-Watson statistic
        dw_stat = durbin_watson(model_.resid)

        # Get the Jarque Bera statistics
        jb_prob = jarque_bera(model_.resid)[1]

        # Verify assumptions of OLS
        if jb_prob < 0.05:
            residual_normality_verified = False
        else:
            residual_normality_verified = True

        # Verify the normality of the residuals. This should be common for OLS and ARIMA
        if not residual_normality_verified:
            print("WARNING: Jarque-Bera statistic = {0}".format(np.round(jb_prob, 3)))
            print('Evidence that residuals do not follow a normal distribution.')
            print('\n')

        # Assume independence of the residuals is satisfied and test this assumption below
        residual_independence_verified = True
        stat_type = None
        residual_independence_violation = None

        # Verify independence of residuals (OLS)
        if model_type == 'interrupted_time_series':

            stat_type = 'Durbin-Waston'
            if dw_stat < 1.5:
                residual_independence_verified = False
                residual_independence_violation = 'positive auto-correlation'
            if dw_stat > 2.5:
                residual_independence_verified = False
                residual_independence_violation = 'negative auto-correlation'

        if model_type in ('naive_arima_interrupted_time_series', 'auto_arima_interrupted_time_series'):

            stat_type = 'Ljung-Box'
            # Check for serial auto-correlation in the residuals
            # Get the Ljung-Box Q statistic
            q_stat_prob = pd.read_html(model_.summary().tables[2].as_html(), index_col=0)[0][1]['Prob(Q):']

            if q_stat_prob < 0.05:
                residual_independence_verified = False
                residual_independence_violation = 'serial auto-correlation'

        if not residual_independence_verified:
            print("WARNING: {0} statistic = {1}".format(stat_type, np.round(dw_stat, 3)))
            print("evidence for {0} in residuals.  Residuals may not be independent. Revise the choice of model".format(residual_independence_violation))
            print("Generating auto-correlation and partial auto-correlation plots for review...")
            print('\n')

            # Generate autocorrelation plot for the model residuals
            file_name_auto_corr = 'autocorrelation_of_residuals_{0}.png'.format(current_time)
            file_name_auto_corr = os.path.join(save_path, file_name_auto_corr)
            sm.graphics.tsa.plot_acf(model_.resid, lags=10, title='Autocorrelation of model residuals')
            plt.savefig(file_name_auto_corr)

            # Generate partial autocorrelation plot for the model residuals
            file_name_partial_auto_corr = 'partial_autocorrelation_of_residuals_{0}.png'.format(current_time)
            file_name_partial_auto_corr = os.path.join(save_path, file_name_partial_auto_corr)
            sm.graphics.tsa.plot_pacf(model_.resid, lags=10, title='Partial Autocorrelation of model residuals')
            plt.savefig(file_name_partial_auto_corr)

        # Make recommendations about what to do
        if stat_type == 'Durbin-Waston' and not residual_independence_verified and model_type == 'interrupted_time_series':
            print("Consider using an ARIMA model instead of the Ordinary Least Squares approach")
            print('\n')
        if stat_type == 'Ljung-Box' and not residual_independence_verified and model_type == 'naive_arima_interrupted_time_series':
            print('Consider using a different set of model parameters')
            print('Examine the auto-correlations and partial auto-correlations in your data to select order parameters or using the auto_arima_interrupted_time_series option.')
            print('\n')
        if stat_type == 'Ljung-Box' and not residual_independence_verified and model_type == 'auto_arima_interrupted_time_series':
            print("Consider re-examining the fit of the auto ARIMA model.")
            print("Increase the number of iterations for the parameter search if necessary, or consider using a different analytical approach")
            print('\n')
        if not residual_normality_verified:
            print("Consider examining the distribution of the model residuals. A QQ-plot of the residuals could also help understanding deviations from normality.")
            print("Normality of the residuals may fail due to sample size issues. Either try and get more data, or consider a different analytical approach.")
            print('\n')

        if not residual_independence_verified or not residual_normality_verified:
            return True
        else:
            return False

    def calculate_counterfactuals(self, df: pd.DataFrame, arima_params_dict: dict, variable_name_dict: dict, model_: Union[sm.regression.linear_model.RegressionResultsWrapper, sms.tsa.arima.model.ARIMAResultsWrapper, sms.tsa.statespace.sarimax.SARIMAXResultsWrapper], model_type: str, save_path: str, figsize: tuple, current_time: str, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Function to calculate the counterfactual provided a model type and data. This will provide model predictions in both the pre- and post-treatment periods as well as
        counterfactual values from the model in the post-treatment period (along with confidence intervals).

        For the simple interrupted_time_series type, this just sets the post-treatment variable (D) and the time since intervention variable (P) to zero and uses the pre-fit model
        to predict the outcome in the post-treatment period. These are the modeled counterfactuals.

        For both the naive_arima_interrupted_time_series and auto_arima_interrupted_time_series methods, a new model using the same parameters as the one fit to estimate the effect
        size will be fit on the pre-treatment period with the post-treatment variable (D) and the time since intervention variable (P) to zero and retrains the model. Forecasts
        from this model on the post-treatment period are then the counterfactual estimates.  This is necessary here as the time series models may rely on previous data points to
        make forecasts.  i.e. forecasting the counterfactual 3 days after the intervention requires knowing what happened 2 days after the intervention. This would not be a
        counterfactual estimate then as the model already knows what happened. Hence, a new model must be fit using the same parameters as the full model.

        :param df: DataFrame with the data you want to investigate. Should be the output from prep_data_for_interrupted_time_series as some column names are assumed
        :param arima_params_dict: Dictionary with parameters to pass to ARIMA models.  naive_arima_interrupted_time_series will look for two keys: order and seasonal_order. Both
                                  are expected to have tuples as items.  See here for a definition of what goes into the tuples:
                                  https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
                                  auto_arima_interrupted_time_series will look for two other keys (and ignore order and seasonal_order); seasonal_auto_arima and
                                  seasonal_period_auto_arima. seasonal_auto_arima is a Boolean. If True, the model will auto-fit a SARIMA model, otherwise will default to ARIMA.
                                  seasonal_period_auto_arima is the periodicity of your data (e.g. 7 days).  Setting this to 1 will override seasonal_auto_arima and only an ARIMA
                                  model will be fit
        :param variable_name_dict: Name of the variables in your DataFrame as keys and how each one maps to one of outcome, T, D, P as values.  See the
                                   prep_data_for_interrupted_time_series method for definitions of these variables. It is recommended to just pre-process your data through
                                   prep_data_for_interrupted_time_series. In this all you need to do is specify the outcome column name.
                                   i.e. variable_name_dict = {'outcome': OUTCOME_COLUMN_NAME_IN_DATA, 'T': 'T', 'D': 'D', 'P': 'P'}
        :param model_: A fit model object. Should be the output of generate_model
        :param model_type: Type of model you want to fit.  See then __init__ method for a list of currently supported model types
        :param figsize: Tuple containing the size of the output figure which plots model predictions and counterfactuals.
        :param save_path: Path to where you want to save any generated figures. If none, the plots will be saved in the current working directory
        :param alpha: The probability of rejecting the null hypothesis when the null hypothesis is true. i.e. 1 - alpha = significance level
        :param current_time: String containing the current datetime. This is used for generating plot filenames.

        :return: Two DataFrames. One containing the estimated counterfactuals, and another the model predictions in the pre- and post-treatment periods (the latter are not
                 counterfactuals as they do not represent the scenario where the intervention never occurred).
        """

        # We need the start and end indices for the post-treatment period
        start = df.query("{0}==1".format(variable_name_dict['D'])).index.astype(int).min()
        end = df.query("{0}==1".format(variable_name_dict['D'])).index.astype(int).max()

        if model_type == 'interrupted_time_series':
            predictions = model_.get_prediction(df)
            summary = predictions.summary_frame(alpha=alpha)

            y_pred = predictions.predicted_mean
            y_pred = pd.DataFrame({'y_hat': y_pred})
            y_pred[variable_name_dict['D']] = df[variable_name_dict['D']].values

            # Make a copy of the original DataFrame
            # This will contain the counterfactual predictions, which assumes no intervention has happened
            # i.e. D = 0 and P = 0 throughout the entire observation period
            df_cf = df.copy()
            df_cf[variable_name_dict['D']] = 0.0
            df_cf[variable_name_dict['P']] = 0.0

            cf = model_.get_prediction(df_cf).summary_frame(alpha=alpha)
            # Restore the true post-treatment variable values so that we can distinguish counterfactual predictions in the actual post treatment period.
            cf[variable_name_dict['D']] = df[variable_name_dict['D']].values

        if model_type in ('naive_arima_interrupted_time_series', 'auto_arima_interrupted_time_series'):
            predictions = model_.get_prediction(0, end)

            arima_params_dict_ = arima_params_dict.copy()
            # Get the order and seasonal order parameters from the original model
            if model_type == 'auto_arima_interrupted_time_series':
                arima_params_dict_['order'] = self.auto_arima_fit_order
                arima_params_dict_['seasonal_order'] = self.auto_arima_fit_seasonal_order

            arima_cf = self.generate_model(df=df[:start],
                                           variable_name_dict=variable_name_dict,
                                           arima_params_dict=arima_params_dict_,
                                           model_type='naive_arima_interrupted_time_series',
                                           refit_=True)
            # Model prediction means
            y_pred = predictions.predicted_mean
            y_pred = pd.DataFrame({'y_hat': y_pred})
            y_pred[variable_name_dict['D']] = df[variable_name_dict['D']].values

            # Counterfactual mean and 95% confidence interval
            # These are now out-of-sample, hence they are forecasts rather than the in-sample predictions from the effect size model above.
            forecast_steps = end - start + 1
            cf = arima_cf.get_forecast(int(forecast_steps), exog=df[variable_name_dict['T']][start:]).summary_frame(alpha=alpha)
            cf[variable_name_dict['D']] = 1

        # Now plot the results. We can create a generic method to handle this in all cases now.
        sig_level = np.round(int((1 - alpha)*100), 2)
        plot_title = "Counterfactual estimate for {0}. Estimated Lift: {1}% - {2}% CI ({3}% - {4}%)".format(self.outcome_name,
                                                                                                            np.round(self.effect_size_rel*100, 3),
                                                                                                            sig_level,
                                                                                                            np.round(self.effect_size_rel_lower*100, 3),
                                                                                                            np.round(self.effect_size_rel_upper*100, 3))
        self.plot_counterfactuals(df=df,
                                  cf=cf,
                                  y_pred=y_pred,
                                  variable_name_dict=variable_name_dict,
                                  scatter_label=self.outcome_name,
                                  title=plot_title,
                                  ylabel=self.outcome_name,
                                  xlabel='Date',
                                  save_path=save_path,
                                  figsize=figsize,
                                  alpha=alpha,
                                  current_time=current_time,
                                  start=start)

        return cf, y_pred

    def plot_counterfactuals(self, df: pd.DataFrame, cf: pd.DataFrame, y_pred: pd.DataFrame, start, variable_name_dict: dict, scatter_label: str, title: str, xlabel: str, ylabel: str, save_path: str, current_time: str, figsize: tuple = (16, 10), alpha: float = 0.1):
        """
        Function to plot the estimated counterfactuals, model predictions, and true observations.  The plot will also contain the estimated relative effect size, and its confidence
        interval at any desired level of significance (default is 90%).

        :param df: DataFrame with the data you want to investigate. Should be the output from prep_data_for_interrupted_time_series as some column names are assumed
        :param cf: DataFrame containing the counterfactual estimates. Should be the first output from the function calculate_counterfactuals
        :param y_pred: DataFrame containing the model predictions. Should be the second output from the function calculate_counterfactuals
        :param variable_name_dict: Name of the variables in your DataFrame as keys and how each one maps to one of outcome, T, D, P as values.  See the
                                   prep_data_for_interrupted_time_series method for definitions of these variables. It is recommended to just pre-process your data through
                                   prep_data_for_interrupted_time_series. In this all you need to do is specify the outcome column name.
                                   i.e. variable_name_dict = {'outcome': OUTCOME_COLUMN_NAME_IN_DATA, 'T': 'T', 'D': 'D', 'P': 'P'}
        :param scatter_label: Legend label for the actual true observations (these are plotted as a scatter plot with the predictions and counterfactuals overlaid)
        :param title: Title for the plot
        :param xlabel: Label for the plot's x-axis
        :param ylabel: Label for the plot' y-axis
        :param save_path: Tuple containing the size of the output figure which plots model predictions and counterfactuals.
        :param figsize: Path to where you want to save any generated figures. If none, the plots will be saved in the current working directory
        :param start: The indiex in the input DataFrame df corresponding to the first time the intervention occurred (i.e. the treatment date)
        :param alpha: The probability of rejecting the null hypothesis when the null hypothesis is true. i.e. 1 - alpha = significance level
        :param current_time: String containing the current datetime. This is used for generating plot filenames.

        :return: Nothing
        """

        # Extract column names
        post_treatment_col = variable_name_dict['D']
        time_counter_variable = variable_name_dict['T']
        outcome_col = variable_name_dict['outcome']

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(df[time_counter_variable], df[outcome_col], facecolors='none', edgecolors='steelblue', label=scatter_label)
        ax.plot(df.query("{0}==1".format(post_treatment_col))[time_counter_variable], y_pred.query("{0}==1".format(post_treatment_col))['y_hat'], 'b-', label='model predictions')
        ax.plot(df.query("{0}==0".format(post_treatment_col))[time_counter_variable], y_pred.query("{0}==0".format(post_treatment_col))['y_hat'], 'b-')

        ax.plot(df.query("{0}==1".format(post_treatment_col))[time_counter_variable], cf.query("{0}==1".format(post_treatment_col))['mean'], 'k.', label='counterfactual')
        ax.fill_between(
            df.query("{0}==1".format(post_treatment_col))[time_counter_variable],
            cf.query("{0}==1".format(post_treatment_col))['mean_ci_lower'],
            cf.query("{0}==1".format(post_treatment_col))['mean_ci_upper'], color='k', alpha=alpha, label='counterfactual 95% CI')
        ax.axvline(x=start, linestyle='--', ymax=1, color='red', label='{0} occurred'.format(self.treatment_name))

        ax.legend(loc='best')
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.title(title)

        # Generate filename
        file_name = 'counterfactual_estimate_for_{0}_{1}.png'.format(self.treatment_name, current_time)
        file_name = os.path.join(save_path, file_name)

        plt.savefig(file_name)









