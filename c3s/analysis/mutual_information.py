import math
import numpy as np
from typing import List, Dict
from scipy.sparse.linalg import expm
from .marginalization import get_Delta_vectors


def mutual_information(P, Deltas, base=2):
    """calculates mutual information between 2 subsets of species for a single time point

    Args:
        P:
            probability vector of len(M)
        Deltas:
        base:
            based used for the logarithm calculation
    Returns:
        mutual_information:
            instantaneous mutual information between X and Y
    """

    _mutual_information = 0
    _mutual_information_terms = {}
    for x, Dx in Deltas.x.items():
        for y, Dy in Deltas.y.items():
            xy = x + y
            if xy not in Deltas.xy:
                # skip cases where the concatenated coordinate tuples
                # were never in the joint distribution to begin with
                continue
            Dxy = Deltas.xy[xy]
            p_xy = np.dot(P, Dxy)
            if p_xy == 0:
                # add zero to the sum if p_xy is 0
                # need to do this because 0*np.log(0) returns an error
                continue
            p_x = np.dot(P, Dx)
            p_y = np.dot(P, Dy)
            this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
            _mutual_information += this_term
            _mutual_information_terms[xy] = this_term

    return _mutual_information, _mutual_information_terms

def _mi_matrix(system, Deltas, base):
    """calculates the pairwise instantaneous mutual information between
    every timepoint in the trajectory"""

    trajectory = system.trajectory
    dt = system._dt
    N_timesteps = len(trajectory)
    global_mi_matrix = np.zeros(shape=(N_timesteps, N_timesteps), dtype=np.float64)
    local_dts = [dt * ts for ts in range(N_timesteps)]
    # using a generator expression to save on memory here
    local_Qs = (expm(system.G*i) for i in local_dts)
    for top_col_loc, Q in zip(range(N_timesteps), local_Qs):
        if top_col_loc == 0:
            diagonal = _mi_diagonal(Deltas, base)
            np.fill_diagonal(global_mi_matrix, diagonal)
            continue
        # first Q is upper triangle, second is for lower triangle
        tildeQ_matrices = {(x,y): (Q*np.outer(Dy, Dx), Q*np.outer(Dx, Dy))
                                  for x, Dx in Deltas.x.items() for y, Dy in Deltas.y.items()}
        # here we calculate the mutual information along the offdiagonal
        # so as to avoid repeatedly computing the matrix exponential
        i =  0
        j = top_col_loc
        while j < N_timesteps:    # the edge of the matrix
            upper, lower = _matrix_element(i, j, Deltas, tildeQ_matrices, base)
            global_mi_matrix[i,j] = upper
            global_mi_matrix[j,i] = lower
            i += 1
            j += 1

    return global_mi_matrix


def _matrix_element(i, j, system, Deltas, tildeQ_matrices, base):
    """calculates every matrix element for symmetric off diagonals"""

    trajectory = system.trajectory
    # here I assume j is always at the later timepoint
    Pi = trajectory[i]
    Pj = trajectory[j]

    mi_sum_upper = 0
    for x, Dx in Deltas.x.items():
        for y, Dy in Deltas.y.items():
            tildeQ = tildeQ_matrices[(x,y)][0]
            p_xy= np.dot(np.dot(tildeQ, Pi), Dy)
            if p_xy == 0:
                # add zero to the sum if p_xy is 0
                # need to do this because 0*np.log(0) returns an error
                continue
            p_x = np.dot(Pi, Dx)
            p_y = np.dot(Pj, Dy)
            this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
            mi_sum_upper += this_term
    mi_sum_lower = 0
    for x, Dx in Deltas.x.items():
        for y, Dy in Deltas.y.items():
            tildeQ = tildeQ_matrices[(x,y)][1]
            p_xy= np.dot(np.dot(tildeQ, Pi), Dx)
            if p_xy == 0:
                # add zero to the sum if p_xy is 0
                # need to do this because 0*np.log(0) returns an error
                continue
            p_x = np.dot(Pj, Dx)
            p_y = np.dot(Pi, Dy)
            this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
            mi_sum_lower += this_term

    return mi_sum_upper, mi_sum_lower
