import numpy as np
from numba import njit
from .sparse_matrix import sm_times_array


@njit
def binary_search(vector, x, low=0, high=None):
    """"""
    if high is None:
        high = len(vector)-1
    if high >= low:
        mid = (high + low) // 2
        if vector[mid] == x:
            return mid
        elif vector[mid] > x:
            return binary_search(vector, x, low=low, high=mid-1)
        else:
            return binary_search(vector, x, low=mid+1, high=high)
    else:
        return -1


#@njit
def cartesian_product(space_1, space_2):
    product_space = [np.concatenate((state_i, state_j)) for state_i in space_1 for state_j in space_2]
    return np.stack(product_space)


#@njit
def combine_state_spaces(*subspaces):
    if len(subspaces) == 2:
        return cartesian_product(subspaces[0], subspaces[1])
    return cartesian_product(subspaces[0], combine_state_spaces(*subspaces[1:]))


#@njit
def generate_subspace(Constraint):
    limit = Constraint.value
    N = len(Constraint.species_involved)
    if Constraint.separator == '<=':
        subspace = np.arange(limit + 1)
        return subspace.reshape(subspace.size, -1)
    elif Constraint.separator == '=':
        subspace = [[n // (limit + 1)**i % (limit + 1) for i in range(N)] for n in range((limit + 1)**N)]
        subspace = np.stack([np.flip(state) for state in subspace if sum(state) == limit])
        return subspace


#@njit
def vector_to_number(vector, N, base):
    try:
        number = (vector * (base**np.arange(N-1, -1, -1))).sum(axis=1)
    except np.AxisError:
        number = (vector * (base ** np.arange(N - 1, -1, -1))).sum()
    return number


@njit
def IMU_timestep(p_0, B, OmegaT):
    """Inverse Marginalized Uniformization"""

    log_poisson_factor = -OmegaT
    poisson_factor_cumulative = np.exp(log_poisson_factor)

    sum_ = p_0 * np.exp(log_poisson_factor)
    k_max = OmegaT + 6 * np.sqrt(OmegaT)
    for k in np.arange(1, k_max):
        log_poisson_factor += np.log(OmegaT / k)
        pf = np.exp(log_poisson_factor)
        poisson_factor_cumulative += pf
        p_0 = sm_times_array(B, p_0)
        sum_ += p_0 * pf

    # guarantees sum_ sums to 1
    sum_ += p_0 * (1.0 - poisson_factor_cumulative)
    return sum_


@njit
def EXPM_timestep(p_0, Q):
    """np.dot() with scipy.sparse.linalg.expm()"""
    return Q.dot(p_0)
