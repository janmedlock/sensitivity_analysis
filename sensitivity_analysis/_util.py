import warnings

import pandas
import statsmodels.api


def model_eval(model, parameters):
    '''Evaluate `model()` with `parameters`,
    handling the cases when `parameters`
    is a `dict()`, `pandas.Series()`, ...'''
    try:
        return model(**parameters)
    except TypeError:
        return model(*parameters)


def add_constant(X):
    if isinstance(X, pandas.DataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    category=FutureWarning,
                                    module='numpy')
            return statsmodels.api.add_constant(X)
    else:
        return statsmodels.api.add_constant(X.T)
