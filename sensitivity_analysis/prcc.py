'''Sensitivity analysis using the partial rank correlation coefficient.'''

from . import sampling
from . import stats
from . import util


def get(model, parameters, n_samples):
    X = sampling.get_Latin_hypercube(parameters, n_samples)
    y = util.model_eval(model, X)
    return stats.prcc(X, y)
