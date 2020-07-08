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


def Sobol_indexes(model, parameters, n_samples, alpha=0.5):
    '''Saltelli et al's algorithm from section 4.6.'''
    A = sampling.samples_Latin_hypercube(parameters, n_samples)
    B = sampling.samples_Latin_hypercube(parameters, n_samples)
    y_A = _util.model_eval(model, A)
    y_B = _util.model_eval(model, B)
    y = numpy.hstack((y_A, y_B))
    y_mean = numpy.mean(y)
    y_var = numpy.var(y, ddof=1)
    if hasattr(parameters, 'keys'):
        index = parameters.keys()
        S = pandas.Series(index=index)
        S_PE = pandas.Series(index=index)
        S_T = pandas.Series(index=index)
        S_T_PE = pandas.Series(index=index)
    else:
        index = range(len(parameters))
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


def S_RBD(model, parameters, n_samples, n_freqs=6):
    '''The algorithm from Saltelli et al, page 168, cleaned up a bit.
    E(y|X_{~i}) is approximated by an `n_freqs`-order Fourier expansion.'''
    s_0 = numpy.linspace(0, 2 * numpy.pi, n_samples)
    if hasattr(parameters, 'keys'):
        index = parameters.keys()
        s = pandas.DataFrame({i: numpy.random.permutation(s_0)
                              for i in index})
        q = numpy.arccos(numpy.cos(s)) / numpy.pi
        X = pandas.DataFrame({i: parameters[i].ppf(q[i])
                              for i in index})
        S = pandas.Series(index=index)
    else:
        index = range(len(parameters))
        s = numpy.row_stack([numpy.random.permutation(s_0)
                             for i in index])
        q = numpy.arccos(numpy.cos(s)) / numpy.pi
        X = numpy.row_stack([parameters[i].ppf(q[i])
                             for i in index])
        S = numpy.empty(len(index))
    y = _util.model_eval(model, X)
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


def S_RBD_DCT(model, parameters, n_samples, n_freqs=6):
    '''RBD using the DCT rather than the FFT.'''
    q_0 = numpy.linspace(0, 1, n_samples)
    if hasattr(parameters, 'keys'):
        index = parameters.keys()
        q = pandas.DataFrame({i: numpy.random.permutation(q_0)
                              for i in index})
        X = pandas.DataFrame({i: parameters[i].ppf(q[i])
                              for i in index})
        S = pandas.Series(index=index)
    else:
        index = range(len(parameters))
        q = numpy.row_stack([numpy.random.permutation(q_0)
                             for i in index])
        X = numpy.row_stack([parameters[i].ppf(q[i])
                             for i in index])
        S = numpy.empty(len(index))
    y = _util.model_eval(model, X)
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
