import pandas
import seaborn

from . import sampling
from .stats import _linear_residuals
from . import _util


def scatterplots_samples(X, y, sharex=False):
    '''Make scatterplots of y against X.'''
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X.T)
    if not isinstance(y, pandas.Series):
        y = pandas.Series(y)
    if y.name is None:
        y.rename('y', inplace=True)
    X_tidy = (X.rename_axis(columns='factor').stack()
               .rename('X').reset_index('factor'))
    data = pandas.concat((X_tidy, y), axis='columns')
    fg = seaborn.lmplot(data=data, x='X', y=y.name, col='factor',
                        hue='factor', ci=None, truncate=False,
                        sharex=sharex,
                        line_kws=dict(color='black',
                                      alpha=0.5))
    fg.set_titles('')
    for (ax, p) in zip(fg.axes.flat, X.columns):
        ax.set_xlabel(p)
    return fg


def scatterplots(model, parameters, n_samples, sharex=False):
    '''Make scatterplots of y against X.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    y = _util.model_eval(model, X)
    return scatterplots_samples(X, y, sharex=sharex)


def residualplots_samples(X, y, sharex=False):
    '''Make scatterplots of residual y against residual X.'''
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X.T)
    if not isinstance(y, pandas.Series):
        y = pandas.Series(y)
    if y.name is None:
        y.rename('y', inplace=True)
    data = []
    for (i, x) in X.items():
        Z = X.drop(columns=i)
        x_resid = _linear_residuals(Z, x)
        y_resid = _linear_residuals(Z, y)
        data.append(pandas.DataFrame({'factor': i,
                                      'x_resid': x_resid,
                                      'y_resid': y_resid}))
    data = pandas.concat(data)
    fg = seaborn.lmplot(data=data, x='x_resid', y='y_resid', col='factor',
                        hue='factor', ci=None, truncate=False,
                        sharex=sharex,
                        line_kws=dict(color='black',
                                      alpha=0.5))
    fg.set_titles('')
    for (ax, p) in zip(fg.axes.flat, X.columns):
        ax.set_xlabel('residual ' + p)
    fg.set_ylabels('residual ' + y.name)
    return fg


def residualplots(model, parameters, n_samples, sharex=False):
    '''Make scatterplots of residual y against residual X.'''
    X = sampling.samples_Latin_hypercube(parameters, n_samples)
    y = _util.model_eval(model, X)
    return residualplots_samples(X, y, sharex=sharex)
