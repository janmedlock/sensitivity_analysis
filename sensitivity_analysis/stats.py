'''Statistical functions.'''

import numpy
import pandas
import scipy.stats
import statsmodels.api


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


def cc(X, y):
    '''Correlation coefficient, i.e. Pearson's rho.'''
    try:
        # If `X` is a pandas.DataFrame().
        rho = X.corrwith(y)
    except AttributeError:
        rho, _ = numpy.apply_along_axis(scipy.stats.pearsonr, 0, X, y)
    return rho


def get_linear_residuals(Z, y):
    '''Find the residual between y and
    the linear regression of y against Z.'''
    Z_ = statsmodels.api.add_constant(Z)
    lm = statsmodels.api.OLS(y, Z_).fit()
    return lm.resid


def pcc(X, y):
    '''Partial correlation coefficient.'''
    X_arr = numpy.asarray(X)
    ncols = X_arr.shape[1]
    rho = numpy.empty(ncols)
    for col in range(ncols):
        x = X_arr[:, col]
        # All of the other columns except `col`.
        Z = X_arr[:, numpy.arange(ncols) != col]
        x_res = get_linear_residuals(Z, x)
        y_res = get_linear_residuals(Z, y)
        rho[col], _ = scipy.stats.pearsonr(x_res, y_res)
    try:
        # If `X` is a pandas.DataFrame().
        rho = pandas.Series(rho, index=X.columns)
    except AttributeError:
        pass
    return rho


def rcc(X, y):
    '''Rank correlation coefficient, i.e. Spearman's rho.'''
    try:
        # If `X` is a pandas.DataFrame().
        rho = X.corrwith(y, method='spearman')
    except AttributeError:
        rho, _ = numpy.apply_along_axis(scipy.stats.spearmanr, 0, X, y)
    return rho


def get_rank(X):
    '''Get quantile rank of X.
    The lowest rank is 0 and the highest is 1.'''
    try:
        # If `X` is a pandas.DataFrame().
        rank = X.rank()
    except AttributeError:
        rank = numpy.apply_along_axis(scipy.stats.rankdata, 0, X)
    return (rank - 1) / (len(X) - 1)


def prcc(X, y):
    '''Partial rank correlation coefficient.'''
    return pcc(get_rank(X), get_rank(y))
