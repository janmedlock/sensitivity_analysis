'''Local sensitivity measures based on derivatives.'''

import numpy
import pandas

from . import sampling
from . import stats
from . import _util


def _gradient(X, model, y=None, epsilon=1e-6):
    '''Estimate the gradient of model at X.'''
    if y is None:
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
    if isinstance(X, pandas.Series):
        X1 = pandas.DataFrame(X1.T, columns=X.index)
    elif isinstance(X, pandas.DataFrame):
        X1 = dict(zip(X.columns, X1))
    y1 = _util.model_eval(model, X1)
    grad = (y1 - numpy.asarray(y)[..., None]) / epsilon
    if isinstance(X, pandas.Series):
        if isinstance(grad, pandas.Series):
            grad.set_axis(X.index, inplace=True)
        else:
            grad = pandas.Series(grad, index=X.index)
    elif isinstance(X, pandas.DataFrame):
        if isinstance(grad, pandas.DataFrame):
            grad.set_axis(X.index, axis='index', inplace=True)
            grad.set_axis(X.columns, axis='columns', inplace=True)
        else:
            grad = pandas.DataFrame(grad, index=X.index, columns=X.columns)
    elif grad.ndim == 2:
        grad = grad.T
    return grad


def sensitivity_samples(X, y, model, normalized=True, _y_mean=False):
    '''The derivatives evaluated at the mean parameter values.'''
    X_mean = stats.mean(X)
    y_mean = _util.model_eval(model, X_mean)
    S = _gradient(X_mean, model, y_mean)
    if normalized:
        S *= stats.std(X) / stats.std(y)
    if _y_mean:
        return (S, y_mean)
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
    (S, y_mean) = sensitivity_samples(X, y, model, normalized=normalized,
                                      _y_mean=True)
    return S * stats.mean(X) / y_mean


def elasticity(model, parameters, n_samples, normalized=True):
    '''The elasticity evaluated at the mean parameter values.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    y = _util.model_eval(model, X)
    return elasticity_samples(X, y, model, normalized=normalized)
