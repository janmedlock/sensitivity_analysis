'''Local sensitivity measures based on derivatives.'''

import numpy
import pandas

from . import sampling
from . import stats
from . import _util


def _gradient(model, X, epsilon=1e-6):
    '''Estimate the gradient of model at X.'''
    y = _util.model_eval(model, X)
    if isinstance(X, pandas.Series):
        X_arr = X.values
    elif isinstance(X, pandas.DataFrame):
        X_arr = X.values.T
    else:
        X_arr = X
    # Add a new dimension to end of X and along this dimension
    # add epsilon each component.
    shape = (len(X_arr), ) + (1, ) * (numpy.ndim(X_arr) - 1) + (len(X_arr), )
    eye = numpy.identity(len(X_arr)).reshape(shape)
    X1 = X_arr[..., None] + epsilon * eye
    y1 = _util.model_eval(model, X1)
    grad = (y1 - numpy.asarray(y)[..., None]) / epsilon
    if isinstance(X, pandas.Series):
        return pandas.Series(grad, index=X.index)
    elif isinstance(X, pandas.DataFrame):
        return pandas.DataFrame(grad, index=X.index, columns=X.columns)
    else:
        return grad.T


def sensitivity_samples(X, y, model, normalized=True):
    '''The derivatives evaluated at the mean parameter values.'''
    S = _gradient(model, stats.mean(X))
    if normalized:
        return S * stats.std(X) / stats.std(y)
    else:
        return S


def sensitivity(model, parameters, n_samples, normalized=True):
    '''The derivatives evaluated at the mean parameter values.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    if normalized:
        y = _util.model_eval(model, X)
    else:
        y = None
    return sensitivity_samples(X, y, model, normalized=normalized)


def elasticity_samples(X, y, model, normalized=True):
    '''The elasticity evaluated at the mean parameter values.'''
    S = sensitivity_samples(X, y, model, normalized=normalized)
    return S * stats.mean(X) / stats.mean(y)


def elasticity(model, parameters, n_samples, normalized=True):
    '''The elasticity evaluated at the mean parameter values.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    y = _util.model_eval(model, X)
    return elasticity_samples(X, y, model, normalized=normalized)
