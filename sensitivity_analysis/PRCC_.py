'''Sensitivity analysis using the partial rank correlation coefficient.'''

from . import sampling
from . import stats
from . import _util


def PRCC_samples(X, y):
    '''The partial rank correlation coefficient.'''
    return stats.PRCC(X, y)


def PRCC(model, parameters, n_samples, seed=None):
    '''The partial rank correlation coefficient.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples, seed=seed)
    y = _util.model_eval(model, X)
    return PRCC_samples(X, y)
