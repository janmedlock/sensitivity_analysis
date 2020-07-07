'''Samplers for random variables.'''

import numpy
import pandas


def get_unstructured(parameters, n):
    '''Get `n` samples from each of `parameters`.'''
    try:
        return pandas.DataFrame(
            {k: p.rvs(n) for (k, p) in parameters.items()})
    except AttributeError:
        return numpy.row_stack(
            [p.rvs(n) for p in parameters])


def _get_structured_1D(parameter, n):
    '''Get `n` structured samples from `parameter`.
    A random sample is taken from each quantile [i / n, (i + 1) / n)
    for i = 0, 1, ..., n - 1.'''
    # Get `n` random variables U_i,
    # each one uniform within the quantile
    # [i / n, (i + 1) / n),
    # for i = 0, 1, ..., n - 1.
    bounds = numpy.linspace(0, 1, n + 1)
    U = numpy.random.uniform(bounds[:-1], bounds[1:])
    # Map the random variables U to random
    # values of `parameter` using the "percent point function",
    # which is the inverse of the CDF.
    samples = parameter.ppf(U)
    # Randomly shuffle the order of the samples.
    return numpy.random.permutation(samples)


def get_Latin_hypercube(parameters, n):
    '''Get `n` samples of `parameters` using Latin hypercube sampling.'''
    try:
        return pandas.DataFrame(
            {k: _get_structured_1D(p, n) for (k, p) in parameters.items()})
    except AttributeError:
        return numpy.row_stack(
            [_get_structured_1D(p, n) for p in parameters])
