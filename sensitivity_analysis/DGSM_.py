'''Derivative-based global sensitivity measure.
Based on Lamboni et al, 2013.'''

import numpy
import pandas

from .local import _gradient
from . import sampling
from . import stats
from . import _util


def DGSM_samples(X, y, model, normalized=False):
    '''The square of the gradient evaluated at sample parameter values.'''
    try:
        D = pandas.DataFrame([_gradient(model, X_i)
                              for (i, X_i) in X.iterrows()])
    except AttributeError:
        D = numpy.column_stack([_gradient(model, X_i)
                                for X_i in zip(*X)])
    v = stats.mean(D ** 2)
    if normalized:
        return v * stats.var(X) / stats.var(y)
    else:
        return v


def DGSM(model, parameters, n_samples, normalized=False):
    '''The square of the gradient evaluated at sample parameter values.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    if normalized:
        y = _util.model_eval(model, X)
    else:
        y = None
    return DGSM_samples(X, y, model, normalized=normalized)
