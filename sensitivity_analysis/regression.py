'''Linear-regression methods.'''

import statsmodels.api

from . import sampling
from . import stats
from . import _util


def linreg_coefficients_samples(X, y, normalized=False):
    '''The linear regression coefficients.'''
    X_ = _util.add_constant(X)
    lm = statsmodels.api.OLS(y, X_).fit()
    beta = lm.params[1:]
    if normalized:
        return beta * stats.std(X) / stats.std(y)
    else:
        return beta


def linreg_coefficients(model, parameters, n_samples, normalized=False):
    '''The linear regression coefficients.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    y = _util.model_eval(model, X)
    return linreg_coefficients_samples(X, y, normalized=normalized)
