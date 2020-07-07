'''Local sensitivity measures based on derivatives.'''

import numpy
import pandas

from . import sampling
from . import stats
from . import util


def gradient(model, X, epsilon=1e-6):
    '''Estimate the gradient of model at X.'''
    y = util.model_eval(model, X)
    # Make n copies of X, then add epsilon to the ith component of each.
    X1 = numpy.tile(numpy.asarray(X)[:, None], len(X))
    X1 += epsilon * numpy.eye(len(X))
    y1 = util.model_eval(model, X1)
    grad = (y1 - y) / epsilon
    if isinstance(X, pandas.Series):
        grad = pandas.Series(grad, index=X.index)
    return grad


def get_sensitivity(model, parameters):
    '''The square of the sigma-normalized derivatives
    evaluated at the mean parameter values.'''
    try:
        X_mean = pandas.Series(
            {i: p.mean() for (i, p) in parameters.items()})
    except AttributeError:
        X_mean = numpy.array(
            [p.mean() for p in parameters])
    S = gradient(model, X_mean)
    return S


def get_sensitivity_sigma_normalized(model, parameters, n_samples):
    '''The square of the sigma-normalized derivatives
    evaluated at the mean parameter values.'''
    X = sampling.get_Latin_hypercube(parameters, n_samples)
    S = gradient(model, stats.mean(X))
    y = util.model_eval(model, X)
    S_sigma = S * stats.std(X) / stats.std(y)
    return S_sigma
