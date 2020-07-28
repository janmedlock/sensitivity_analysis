'''Samplers for random variables.'''

import numpy
import pandas


def samples_unstructured(parameters, n_samples, seed=None):
    '''Get samples from each of `parameters`.'''
    rng = numpy.random.default_rng(seed)
    try:
        (keys, values) = zip(*parameters.items())
    except AttributeError:
        (keys, values) = (None, parameters)
    samples = (p.rvs(n_samples, random_state=rng)
               for p in values)
    if keys is not None:
        return pandas.DataFrame(dict(zip(keys, samples)))
    else:
        return numpy.row_stack(tuple(samples))


def _samples_structured_1D(parameter, n_samples, rng):
    '''Get structured samples from `parameter`.
    A random sample is taken from each quantile
    [i / n_samples, (i + 1) / n_samples)
    for i = 0, 1, ..., n_samples - 1.'''
    # Get random variables U_i,
    # each one uniform within the quantile
    # [i / n_samples, (i + 1) / n_samples),
    # for i = 0, 1, ..., n_samples - 1.
    bounds = numpy.linspace(0, 1, n_samples + 1)
    U = rng.uniform(bounds[:-1], bounds[1:])
    # Map the random variables U to random
    # values of `parameter` using the "percent point function",
    # which is the inverse of the CDF.
    samples = parameter.ppf(U)
    # Randomly shuffle the order of the samples.
    return rng.permutation(samples)


def samples_Latin_hypercube(parameters, n_samples, seed=None):
    '''Get samples of `parameters` using Latin hypercube sampling.'''
    rng = numpy.random.default_rng(seed)
    try:
        (keys, values) = zip(*parameters.items())
    except AttributeError:
        (keys, values) = (None, parameters)
    samples = (_samples_structured_1D(p, n_samples, rng)
               for p in values)
    if keys is not None:
        return pandas.DataFrame(dict(zip(keys, samples)))
    else:
        return numpy.row_stack(tuple(samples))
