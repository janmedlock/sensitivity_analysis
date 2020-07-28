'''Linear-regression methods.'''

import statsmodels.api

from . import sampling
from . import stats
from . import _util


def linreg_coefficients_samples(X, y, normalized=True):
    '''The linear regression coefficients.'''
    X_ = _util.add_constant(X)
    lm = statsmodels.api.OLS(y, X_).fit()
    beta = lm.params[1:]
    if normalized:
        beta *= stats.std(X) / stats.std(y)
    return beta


def linreg_coefficients(model, parameters, n_samples,
                        normalized=True, seed=None):
    '''The linear regression coefficients.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples, seed=seed)
    y = _util.model_eval(model, X)
    return linreg_coefficients_samples(X, y, normalized=normalized)
