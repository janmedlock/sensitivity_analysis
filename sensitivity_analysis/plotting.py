import pandas
import seaborn

from . import sampling
from . import util


def scatterplots(model, parameters, n_samples):
    X = sampling.get_unstructured(parameters, n_samples)
    y = util.model_eval(model, X)
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X.T)
    if not isinstance(y, pandas.Series):
        y = pandas.Series(y)
    X_tidy = (X.rename_axis(columns='factor').stack()
               .rename('X').reset_index('factor'))
    data = pandas.concat((X_tidy, y.rename('y')), axis='columns')
    fg = seaborn.lmplot(data=data, x='X', y='y', col='factor',
                        hue='factor', ci=None, truncate=False,
                        scatter_kws=dict(s=15),
                        line_kws=dict(color='black',
                                      alpha=0.5))
    fg.set_titles('')
    for (ax, p) in zip(fg.axes.flat, parameters.keys()):
        ax.set_xlabel(p)
    return fg
