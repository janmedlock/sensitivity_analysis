'''Derivative-based global sensitivity measure.
Based on Lamboni et al, 2013.'''

from .local import _gradient
from . import sampling
from . import stats
from . import _util


def DGSM_samples(X, y, model, normalized=True):
    '''The square of the gradient evaluated at sample parameter values.'''
    D = _gradient(X, model, y)
    v = stats.mean(D ** 2)
    if normalized:
        v *= stats.var(X) / stats.var(y)
    return v


def DGSM(model, parameters, n_samples, normalized=True, seed=None):
    '''The square of the gradient evaluated at sample parameter values.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples, seed=seed)
    y = _util.model_eval(model, X)
    return DGSM_samples(X, y, model, normalized=normalized)
