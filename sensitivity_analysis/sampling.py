'''Samplers for random variables.'''

import numpy
import pandas


def samples_unstructured(parameters, n_samples):
    '''Get samples from each of `parameters`.'''
    try:
        return pandas.DataFrame(
            {k: p.rvs(n_samples) for (k, p) in parameters.items()})
    except AttributeError:
        return numpy.row_stack(
            [p.rvs(n_samples) for p in parameters])


def _samples_structured_1D(parameter, n_samples):
    '''Get structured samples from `parameter`.
    A random sample is taken from each quantile
    [i / n_samples, (i + 1) / n_samples)
    for i = 0, 1, ..., n_samples - 1.'''
    # Get random variables U_i,
    # each one uniform within the quantile
    # [i / n_samples, (i + 1) / n_samples),
    # for i = 0, 1, ..., n_samples - 1.
    bounds = numpy.linspace(0, 1, n_samples + 1)
    U = numpy.random.uniform(bounds[:-1], bounds[1:])
    # Map the random variables U to random
    # values of `parameter` using the "percent point function",
    # which is the inverse of the CDF.
    samples = parameter.ppf(U)
    # Randomly shuffle the order of the samples.
    return numpy.random.permutation(samples)


def samples_Latin_hypercube(parameters, n_samples):
    '''Get samples of `parameters` using Latin hypercube sampling.'''
    try:
        return pandas.DataFrame({k: _samples_structured_1D(p, n_samples)
                                 for (k, p) in parameters.items()})
    except AttributeError:
        return numpy.row_stack([_samples_structured_1D(p, n_samples)
                                for p in parameters])
