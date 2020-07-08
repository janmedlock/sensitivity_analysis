'''Derivative-based global sensitivity measure.
Based on Lamboni et al, 2013.'''

import numpy
import pandas

from . import local
from . import sampling
from . import stats
from . import util


def get_DGSM(model, parameters, n_samples):
    '''The square of the derivatives
    evaluated at sample parameter values.'''
    X = sampling.get_Latin_hypercube(parameters, n_samples)
    try:
        D = pandas.DataFrame([local.gradient(model, X_i)
                              for (i, X_i) in X.iterrows()])
    except AttributeError:
        D = numpy.column_stack([local.gradient(model, X_i)
                                for X_i in zip(*X)])
    v = stats.mean(D ** 2)
    return v


def get_DGSM_sigma_normalized(model, parameters, n_samples):
    '''The square of the derivatives
    evaluated at sample parameter values,
    normalized by the variances.'''
    X = sampling.get_Latin_hypercube(parameters, n_samples)
    try:
        D = pandas.DataFrame([local.gradient(model, X_i)
                              for (i, X_i) in X.iterrows()])
    except AttributeError:
        D = numpy.column_stack([local.gradient(model, X_i)
                                for X_i in zip(*X)])
    v = stats.mean(D ** 2)
    y = util.model_eval(model, X)
    v_sigma = v * stats.var(X) / stats.var(y)
    return v_sigma
