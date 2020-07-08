'''Sensitivity analysis using the partial rank correlation coefficient.'''

from . import sampling
from . import stats
from . import _util


def PRCC(model, parameters, n_samples):
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    y = _util.model_eval(model, X)
    return stats.PRCC(X, y)
