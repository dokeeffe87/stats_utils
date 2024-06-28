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

    def __init__(self):
        self.supported_model_types = ('interrupted_time_series', 'naive_arima_interrupted_time_series', 'auto_arima_interrupted_time_series', 'regression_discontinuity')
        self.interrupted_time_series_vars = ('outcome', 'T', 'D', 'P')

    def interrupted_time_series(self):
        # TODO: this will be a general function to run the entire analysis. It should take a model type, data, and column definition + plot preferences (and saving preferences) and run everything
        pass


