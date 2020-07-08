'''Local sensitivity measures based on derivatives.'''

import numpy
import pandas

from . import sampling
from . import stats
from . import _util


def _gradient(model, X, epsilon=1e-6):
    '''Estimate the gradient of model at X.'''
    y = _util.model_eval(model, X)
    # Make n copies of X, then add epsilon to the ith component of each.
    X1 = numpy.tile(numpy.asarray(X)[:, None], len(X))
    X1 += epsilon * numpy.eye(len(X))
    y1 = _util.model_eval(model, X1)
    grad = (y1 - y) / epsilon
    if isinstance(X, pandas.Series):
        grad = pandas.Series(grad, index=X.index)
    return grad


def sensitivity(model, parameters, n_samples, normalized=False):
    '''The derivatives evaluated at the mean parameter values.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    S = _gradient(model, stats.mean(X))
    if normalized:
        y = _util.model_eval(model, X)
        return S * stats.std(X) / stats.std(y)
    else:
        return S


def elasticity(model, parameters, n_samples):
    '''The elasticity evaluated at the mean parameter values.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    S = _gradient(model, stats.mean(X))
    y = _util.model_eval(model, X)
    E = S * stats.mean(X) / stats.mean(y)
    return E
