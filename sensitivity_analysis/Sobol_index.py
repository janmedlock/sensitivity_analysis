r'''Sobol' indexes.
The first-order index is
$$S_i =
 \frac{\mathrm{V}_{X_i}\big(\mathrm{E}_{X_{\sim i}}(y|X_i)\big)}
{\mathrm{V}(y)},$$
and the total-order index is
$$S_{\mathrm{T}i} =
 \frac{\mathrm{E}_{X_{\sim i}}\big(\mathrm{V}_{X_i}(y|X_{\sim i})\big)}
{\mathrm{V}(y)}.$$
'''

import numpy
import pandas
import scipy.fft
import scipy.stats

from . import sampling
from . import _util


def _probable_error(z, alpha=0.5):
    '''Find delta such that
    Prob{|mean(z) - mu| > delta} = alpha,
    where mu is the true mean of z.'''
    return (scipy.stats.norm.isf(alpha / 2) / numpy.sqrt(len(z))
            * numpy.sqrt(numpy.mean(z ** 2) - numpy.mean(z) ** 2))


def Sobol_indexes(model, parameters, n_samples, alpha=0.5, seed=None):
    '''Saltelli et al's algorithm from section 4.6.'''
    A = sampling.samples_Latin_hypercube(parameters, n_samples, seed=seed)
    B = sampling.samples_Latin_hypercube(parameters, n_samples, seed=seed)
    y_A = _util.model_eval(model, A)
    y_B = _util.model_eval(model, B)
    y = numpy.hstack((y_A, y_B))
    y_mean = numpy.mean(y)
    y_var = numpy.var(y, ddof=1)
    try:
        index = parameters.keys()
    except AttributeError:
        index = range(len(parameters))
        has_keys = False
    else:
        has_keys = True
    if has_keys:
        S = pandas.Series(index=index)
        S_PE = pandas.Series(index=index)
        S_T = pandas.Series(index=index)
        S_T_PE = pandas.Series(index=index)
    else:
        S = numpy.empty(len(index))
        S_PE = numpy.empty(len(index))
        S_T = numpy.empty(len(index))
        S_T_PE = numpy.empty(len(index))
    for i in index:
        C = B.copy()
        C[i] = A[i]
        y_C = _util.model_eval(model, C)
        y_A_times_y_C = y_A * y_C
        S[i] = (numpy.mean(y_A_times_y_C) - y_mean ** 2) / y_var
        S_PE[i] = _probable_error(y_A_times_y_C, alpha=alpha) / y_var
        y_B_times_y_C = y_B * y_C
        S_T[i] = 1 - (numpy.mean(y_B_times_y_C) - y_mean ** 2) / y_var
        S_T_PE[i] = _probable_error(y_B_times_y_C, alpha=alpha) / y_var
    return (S, S_PE, S_T, S_T_PE)


def S_RBD(model, parameters, n_samples, n_freqs=6, seed=None):
    '''The algorithm from Saltelli et al, page 168, cleaned up a bit.
    E(y|X_{~i}) is approximated by an `n_freqs`-order Fourier expansion.'''
    rng = numpy.random.default_rng(seed)
    s_0 = numpy.linspace(0, 2 * numpy.pi, n_samples)
    try:
        index = parameters.keys()
    except AttributeError:
        index = range(len(parameters))
        has_keys = False
    else:
        has_keys = True
    s = (rng.permutation(s_0)
         for _ in index)
    if has_keys:
        s = pandas.DataFrame(dict(zip(index, s)))
    else:
        s = numpy.row_stack(tuple(s))
    q = numpy.arccos(numpy.cos(s)) / numpy.pi
    X = (parameters[i].ppf(q[i])
         for i in index)
    if has_keys:
        X = pandas.DataFrame(dict(zip(index, X)))
    else:
        X = numpy.row_stack(tuple(X))
    y = _util.model_eval(model, X)
    if has_keys:
        S = pandas.Series(index=index)
    else:
        S = numpy.empty(len(index))
    for i in index:
        order = numpy.argsort(s[i])
        y_reordered = y[order]
        spectrum_y = (2
                      * numpy.abs(numpy.fft.rfft(y_reordered) / n_samples)
                      ** 2)
        # Estimate E(y | X_{~i}) using the `n-freqs`-order Fourier expansion.
        spectrum_EyXnoti = spectrum_y[: n_freqs + 1]
        # Variances are calculated from the Fourier coefficients by
        # Parseval's Theorem.
        var_EyXnoti = spectrum_EyXnoti[1:].sum()
        var_y = spectrum_y[1:].sum()
        S[i] = var_EyXnoti / var_y
    return S


def S_RBD_DCT(model, parameters, n_samples, n_freqs=6, seed=None):
    '''RBD using the DCT rather than the FFT.'''
    rng = numpy.random.default_rng(seed)
    q_0 = numpy.linspace(0, 1, n_samples)
    try:
        index = parameters.keys()
    except AttributeError:
        index = range(len(parameters))
        has_keys = False
    else:
        has_keys = True
    q = (rng.permutation(q_0)
         for _ in index)
    if has_keys:
        q = pandas.DataFrame(dict(zip(index, q)))
    else:
        q = numpy.row_stack(tuple(q))
    X = (parameters[i].ppf(q[i])
         for i in index)
    if has_keys:
        X = pandas.DataFrame(dict(zip(index, X)))
    else:
        X = numpy.row_stack(tuple(X))
    y = _util.model_eval(model, X)
    if has_keys:
        S = pandas.Series(index=index)
    else:
        S = numpy.empty(len(index))
    for i in index:
        order = numpy.argsort(q[i])
        # spectrum_y = scipy.fft.dct(y[order], norm='ortho') ** 2
        # Avoid warning.
        spectrum_y = scipy.fft.dct(numpy.asarray(y[order]), norm='ortho') ** 2
        # Estimate E(y | X_{~i}) using the `n-freqs`-order cosine expansion.
        spectrum_EyXnoti = spectrum_y[: n_freqs + 1]
        # Variances are calculated from the Fourier coefficients by
        # Parseval's Theorem.
        var_EyXnoti = spectrum_EyXnoti[1:].sum() / n_samples
        var_y = spectrum_y[1:].sum() / n_samples
        S[i] = var_EyXnoti / var_y
    return S
