def model_eval(model, parameters):
    '''Evaluate `model()` with `parameters`,
    handling the cases when `parameters`
    is a `dict()`, `pandas.Series()`, ...'''
    try:
        return model(**parameters)
    except TypeError:
        return model(*parameters)
