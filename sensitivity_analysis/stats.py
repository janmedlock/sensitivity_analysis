'''Statistical functions.'''

import numpy
import pandas
import scipy.stats
import statsmodels.api

from . import _util


def mean(X):
    if isinstance(X, (pandas.Series, pandas.DataFrame)):
        return X.mean()
    else:
        return numpy.mean(X, axis=-1)


def std(X, ddof=1):
    if isinstance(X, (pandas.Series, pandas.DataFrame)):
        return X.std(ddof=ddof)
    else:
        return numpy.std(X, ddof=ddof, axis=-1)


def var(X, ddof=1):
    if isinstance(X, (pandas.Series, pandas.DataFrame)):
        return X.var(ddof=ddof)
    else:
        return numpy.var(X, ddof=ddof, axis=-1)


def CC(X, y):
    '''Correlation coefficient, i.e. Pearson's rho.'''
    try:
        # If `X` is a pandas.DataFrame().
        rho = X.corrwith(y)
    except AttributeError:
        rho, _ = numpy.apply_along_axis(scipy.stats.pearsonr, 0, X, y)
    return rho


def _linear_residuals(Z, y):
    '''Find the residual between y and
    the linear regression of y against Z.'''
    Z_ = _util.add_constant(Z)
    lm = statsmodels.api.OLS(y, Z_).fit()
    return lm.resid


def PCC(X, y):
    '''Partial correlation coefficient.'''
    if isinstance(X, pandas.DataFrame):
        index = X.columns
        rho = pandas.Series(index=index)
    else:
        index = pandas.RangeIndex(len(X))
        rho = numpy.empty(len(index))
    for i in index:
        x = X[i]
        # All of the other columns except `i`.
        Z = X[index.drop(i)]
        x_res = _linear_residuals(Z, x)
        y_res = _linear_residuals(Z, y)
        rho[i], _ = scipy.stats.pearsonr(x_res, y_res)
    return rho


def RCC(X, y):
    '''Rank correlation coefficient, i.e. Spearman's rho.'''
    try:
        # If `X` is a pandas.DataFrame().
        rho = X.corrwith(y, method='spearman')
    except AttributeError:
        rho, _ = numpy.apply_along_axis(scipy.stats.spearmanr, 0, X, y)
    return rho


def rank(X):
    '''Get quantile rank of X.
    The lowest rank is 0 and the highest is 1.'''
    try:
        # If `X` is a pandas.DataFrame().
        r = X.rank()
    except AttributeError:
        r = numpy.apply_along_axis(scipy.stats.rankdata, 0, X)
    return (r - 1) / (len(X) - 1)


def PRCC(X, y):
    '''Partial rank correlation coefficient.'''
    return PCC(rank(X), rank(y))
