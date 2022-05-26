import numpy as np
from tqdm.autonotebook import tqdm
from .utils import timeit
from mpi4py import MPI


def calc_mutual_information_old(system, X, Y, timestep='all'):
    """Calculates the mutual information.

    Parameters
    ----------
    system :
        Specifies system.
    X : list
        List of strings that specifies the molecular species of set X.
    Y : list
        List of strings that specifies the molecular species of set Y.
    timestep : str or int, default='all'
        Specifies the timesteps of system.results for which to calculate the mutual information

    Returns
    -------
    mutual_information : flaot
        Calculated mutual information between the marginal distributions given by sets X and Y for
        every timestep specficied.

    """

    X_indices = [system.species.index(x) for x in sorted(X)]
    Y_indices = [system.species.index(y) for y in sorted(Y)]

    constitutive_states = np.array(system.constitutive_states)
    x_states = constitutive_states[:, X_indices]
    y_states = constitutive_states[:, Y_indices]
    assert len(x_states) == len(y_states)
    n_states = len(x_states)

    X_set = []
    for vector in x_states:
        if list(vector) not in X_set:
            X_set.append(list(vector))
    X_set = np.array(X_set)

    Y_set = []
    for vector in y_states:
        if list(vector) not in Y_set:
            Y_set.append(list(vector))
    Y_set = np.array(Y_set)

    def _single_term(x, y):
        p_xy = 0
        p_x = 0
        p_y = 0
        for i in range(n_states):
            if np.array_equal(x, x_states[i]) and np.array_equal(y, y_states[i]):
                p_xy += P[i]
            if np.array_equal(x, x_states[i]):
                p_x += P[i]
            if np.array_equal(y, y_states[i]):
                p_y += P[i]
        if p_xy == 0:
            MI_term = 0
        else:
            MI_term = p_xy * np.log(p_xy / (p_x * p_y))

        return MI_term

    with timeit() as calculation_time:
        if timestep == 'all':
            n_timesteps = len(system.results)
            mutual_information_array = np.empty(shape=n_timesteps)
            for ts in tqdm(range(n_timesteps), leave=True, desc='calculating mutual information'):
                P = system.results[ts]
                mutual_information = 0
                for x in X_set:
                    for y in Y_set:
                        term = _single_term(x, y)
                        mutual_information += term
                mutual_information_array[ts] = mutual_information

    system.timings['t_calculate_mutual_information'] = calculation_time.elapsed

    return mutual_information_array


def calc_mutual_information_new(system, X, Y, timestep='all'):
    """Calculates the mutual information.

        Parameters
        ----------
        system :
            Specifies system.
        X : list of strings
            specifies the unique molecular species of set X.
        Y : list of strings
            specifies the unique molecular species of set Y.
        timestep : str or int, default='all'
            Specifies the timesteps of system.results for which to calculate the mutual information

        Returns
        -------
        mutual_information : array of floats
            Calculated mutual information between the marginal distributions given by sets X and Y for
            every timestep specficied.

        """

    # obtain the indices corresponding to each marginal distribution
    marginal_distribution_in_X = _get_marginal_distribution(system, X)
    marginal_distribution_in_Y = _get_marginal_distribution(system, Y)
    marginal_distribution_in_X_and_Y = _get_marginal_distribution(system, X + Y )

    if timestep == 'all':
        n_timesteps = len(system.results)
        mutual_information_array = np.empty(shape=n_timesteps)
        with timeit() as calculation_time:
            for ts in tqdm(range(n_timesteps), leave=True, desc='calculating mutual information'):
                P = system.results[ts]
                # add each term's value to this total
                mutual_information = 0
                # iterate through every combination in the sum
                for x in marginal_distribution_in_X:
                    for y in marginal_distribution_in_Y:
                        term = _compute_term(P, x, y, marginal_distribution_in_X_and_Y)
                        mutual_information += term
                # store this timestep's value
                mutual_information_array[ts] = mutual_information

    system.timings['t_calculate_mutual_information'] = calculation_time.elapsed

    return mutual_information_array


def _get_marginal_distribution(system, molecular_species):
    """Constructs a marginal probability distribution for specified molecular species."""

    # make a copy as an array for indexing
    constitutive_states = np.array(system.constitutive_states)
    # get the indices of the molecular population vector that correspond to the
    # molecular species we want to generate a distribution over
    indices = [system.species.index(i) for i in sorted(molecular_species)]
    n_subset = constitutive_states[:, indices]

    # each element in this list will contain a tuple of indices that correspond to a degenerate
    # domain point of the joint distribution
    marginal_distribution = []

    curr_state = 0
    states_accounted_for = []
    while len(states_accounted_for) < len(constitutive_states):
        # find the indices of degenerate domain points, start with first index
        if curr_state in states_accounted_for:
            curr_state += 1
            # skip iteration
            continue
        subset_vector = n_subset[curr_state]
        ids = np.argwhere(np.all(n_subset == subset_vector, axis=1)).transpose().flatten().tolist()
        distribution_point = (subset_vector.tolist(), ids)
        marginal_distribution.append(distribution_point)
        states_accounted_for += ids
        # move to the next state and continue
        curr_state += 1

    return marginal_distribution


def _compute_term(P, x, y, xy):
    """Calculates a single term in the mutual information sum.

    term = p(x,y) * log p(x,y) / [p(x)p(y)]

    """

    x_vector = x[0]
    y_vector = y[0]
    idx, idy = x[1], y[1]
    xy_vector = x_vector + y_vector
    possible_xy_vectors = [point[0] for point in xy]
    if xy_vector not in possible_xy_vectors:
        return 0
    idxy = xy[possible_xy_vectors.index(xy_vector)][1]
    p_x = np.sum(P[idx])
    p_y = np.sum(P[idy])
    p_xy = np.sum(P[idxy])

    if p_xy == 0:
        return 0
    else:
        return p_xy * np.log(p_xy / (p_x * p_y))


def calculate_mutual_information_parallel(system, X, Y, timestep='all'):
    pass