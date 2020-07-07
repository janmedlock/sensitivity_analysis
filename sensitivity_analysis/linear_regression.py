'''Linear-regression methods.'''

import numpy
import pandas
import statsmodels.api

from . import sampling
from . import stats
from . import util


def get_coefficients(model, parameters, n_samples):
    '''The square of the standardized regression coefficients.'''
    X = sampling.get_Latin_hypercube(parameters, n_samples)
    y = util.model_eval(model, X)
    if isinstance(X, pandas.DataFrame):
        X_ = statsmodels.api.add_constant(X)
    else:
        X_ = statsmodels.api.add_constant(X.T)
    lm = statsmodels.api.OLS(y, X_).fit()
    beta = lm.params[1:]
    return beta


def get_coefficients_standardized(model, parameters, n_samples):
    '''The square of the standardized regression coefficients.'''
    X = sampling.get_Latin_hypercube(parameters, n_samples)
    y = util.model_eval(model, X)
    if isinstance(X, pandas.DataFrame):
        X_ = statsmodels.api.add_constant(X)
    else:
        X_ = statsmodels.api.add_constant(X.T)
    lm = statsmodels.api.OLS(y, X_).fit()
    beta = lm.params[1:]
    beta_sigma = beta * stats.std(X) / stats.std(y)
    return beta_sigma
