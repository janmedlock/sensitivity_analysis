#!/usr/bin/python3

import joblib
import numpy
from scipy import stats

import sensitivity_analysis


def get_R_0(beta, gamma, mu, N):
    return beta / (gamma + mu)


def initial_state(beta, gamma, mu, N_0, rng):
    '''Generate a random initial `state`.'''
    R_0 = get_R_0(beta, gamma, mu, N_0)
    if R_0 > 1:
        # The endemic equilibrium.
        equilibrium = numpy.array((
            1 / R_0 * N_0,                   # S
            mu / beta * (R_0 - 1) * N_0,     # I
            gamma / beta * (R_0 - 1) * N_0,  # R
        ))
    else:
        # The disease-free equilibrium.
        equilibrium = numpy.array((
            N_0,  # S
            0,    # I
            0,    # R
        ))
    # Fraction in each state.
    p = equilibrium / equilibrium.sum()
    # If R_0 > 1, resample to get I > 0.
    while True:
        state = rng.multinomial(N_0, p)
        (S, I, R) = state
        if (R_0 <= 1) or (I > 0):
            break
    return state


# The transitions that can occur and how they change `state`.
transitions = numpy.array((
    (+1, 0, 0),   # Birth
    (-1, 0, 0),   # Death of S
    (0, -1, 0),   # Death of I
    (0, 0, -1),   # Death of R
    (-1, +1, 0),  # Infection
    (0, -1, +1),  # Recovery
))


def update_hazards(hazards, t, state, beta, gamma, mu):
    '''Update `hazards` for the current state and time.'''
    (S, I, R) = state
    N = state.sum()
    # `hazards` must be in the same order as `transitions`.
    hazards[:] = (
        mu * N,            # Birth
        mu * S,            # Death of S
        mu * I,            # Death of I
        mu * R,            # Death of R
        beta * I / N * S,  # Infection
        gamma * I,         # Recovery
    )
    assert (hazards >= 0).all()


def stop(t, state):
    '''The stopping condition for the simulation.'''
    (S, I, R) = state
    return (I == 0)


def get_persistence_time(beta, gamma, mu, N_0=1000, t_max=10e6, seed=None):
    '''Simulate the persistence time for a stochastic general SIR model.'''
    rng = numpy.random.default_rng(seed)
    t = 0
    state = initial_state(beta, gamma, mu, N_0, rng)
    # Build empty vectors that will get updated in each step.
    n_transitions = len(transitions)
    hazards = numpy.empty(n_transitions, dtype=float)
    hazards_scaled = numpy.empty(n_transitions, dtype=float)
    while (t < t_max) and not stop(t, state):
        update_hazards(hazards, t, state, beta, gamma, mu)
        # Find the time to the next event.
        hazard_total = hazards.sum()
        t += rng.exponential(1 / hazard_total)
        if t < t_max:
            # Find which of the events occurred.
            # Scale the hazards so that they sum to 1.
            hazards_scaled[:] = hazards / hazard_total
            which = rng.choice(n_transitions, p=hazards_scaled)
            state += transitions[which]
        else:
            t = t_max
    return t


def gamma_from_mean_and_shape(mean, shape):
    '''Generate a gamma random variable with the given mean and shape.'''
    scale = mean / shape
    return stats.gamma(shape, scale=scale)


def get_seed(seed_seq):
    '''Helper function to spawn a single seed
    from a `numpy.random.SeedSequence()`.'''
    return seed_seq.spawn(1)[0]


def run_many(model, samples, seed=None, n_jobs=-1):
    '''Run `model` for each of the `samples` parameter sets in parallel.'''
    if isinstance(seed, numpy.random.SeedSequence):
        seed_seq = seed
    else:
        seed_seq = numpy.random.SeedSequence(seed)
    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        try:
            return parallel(
                joblib.delayed(model)(**sample, seed=get_seed(seed_seq))
                for (_, sample) in samples.iterrows())
        except AttributeError:
            return parallel(
                joblib.delayed(model)(*sample, seed=get_seed(seed_seq))
                for sample in samples)


def get_parameters():
    parameters = dict(
        beta=gamma_from_mean_and_shape(0.03, 4),
        gamma=gamma_from_mean_and_shape(0.01, 4),
        mu=gamma_from_mean_and_shape(0.001, 4),
    )
    return parameters


if __name__ == '__main__':
    parameters = get_parameters()
    n_samples = 1000
    seed_seq = numpy.random.SeedSequence(1)
    samples = sensitivity_analysis.samples_Latin_hypercube(
        parameters, n_samples, seed=get_seed(seed_seq))
    numpy.save('samples.npy', samples)
    persistence_times = run_many(get_persistence_time, samples,
                                 seed=get_seed(seed_seq))
    numpy.save('persistence.npy', persistence_times)
