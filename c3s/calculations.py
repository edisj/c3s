import numpy as np
from scipy.linalg import expm
from .simulators import MasterEquation, MPI_ON
from .utils import timeit, ProgressBar, slice_tasks_for_parallel_workers
from typing import List, Dict, Union
from random import shuffle
if MPI_ON:
    from mpi4py import MPI


def calc_mutual_information(system: MasterEquation,
                            X: List[str], Y: List[str],
                            timestep: Union[str, int] = 'all') -> np.ndarray:
    """Calculates the mutual information."""

    if timestep != 'all':
        raise ValueError("other timestep values not implemented yet..")

    n_timesteps = len(system.results)

    with timeit() as get_point_mappings_block:
        x_map = _get_point_mappings(system, X)
        y_map = _get_point_mappings(system, Y)
        xy_map = _get_point_mappings(system, X + Y)

    if system.parallel:
        slices = slice_tasks_for_parallel_workers(n_tasks=n_timesteps, n_workers=system.size)
        sendcounts = tuple(slices[i].stop - slices[i].start for i in range(system.size))
        displacements = tuple(slice_.start for slice_ in slices)
        start = slices[system.rank].start
        stop = slices[system.rank].stop
        local_blocksize = stop - start
        # local buffer that only this process sees
        mutual_information_local = np.empty(shape=local_blocksize, dtype=float)
        # buffer that each process will fill in by receiving data from every other process
        mutual_information_global = np.empty(shape=n_timesteps, dtype=float)
    else:
        start = 0
        stop = n_timesteps
        local_blocksize = stop - start
        mutual_information_local = mutual_information_global = np.empty(shape=local_blocksize, dtype=float)

    with timeit() as calculation_block:
        for i, ts in enumerate(ProgressBar(range(start,stop), position=system.rank,
                                           desc=f'rank {system.rank} calculating mutual information.')):
            probability_vector = system.results[ts]
            # initialize the sum to 0
            mutual_information = 0
            # iterate through every term in the sum and add to the total
            for x_point in x_map:
                for y_point in y_map:
                    xy_point = x_point + y_point
                    if xy_point not in xy_map:
                        # skip cases where the concatenated coordinate tuples
                        # were never in the joint distribution to begin with
                        continue
                    idx = x_map[x_point]
                    idy = y_map[y_point]
                    idxy = xy_map[xy_point]
                    p_xy = np.sum(probability_vector[idxy])
                    if p_xy == 0:
                        # add zero to the sum if p_xy is 0
                        # need to do this because 0*np.log(0) returns an error
                        continue
                    p_x = np.sum(probability_vector[idx])
                    p_y = np.sum(probability_vector[idy])
                    this_term = p_xy * np.log(p_xy / (p_x * p_y))
                    # add any nonzero value to the sum
                    mutual_information += this_term

            mutual_information_local[i] = mutual_information

        if system.parallel:
            system.comm.Allgatherv(sendbuf=mutual_information_local,
                                   recvbuf=(mutual_information_global, sendcounts, displacements, MPI.DOUBLE))

    system.timings['t_get_point_mappings'] = get_point_mappings_block.elapsed
    system.timings['t_calculate_mutual_information'] = calculation_block.elapsed

    return mutual_information_global


def calc_mutual_information_matrix(system: MasterEquation, X: List[str], Y: List[str]) -> np.ndarray:
    """Calculates the mutual information between every pair of timepoints."""

    n_timesteps = len(system.results)
    dts = None
    len_main_diagonal = len(system.results)

    x_map = _get_point_mappings(system, X)
    y_map = _get_point_mappings(system, Y)
    xy_map = _get_point_mappings(system, X + Y)

    #if system.rank == 0:
    dts = [system.dt*ts for ts in range(n_timesteps)]
    print(dts)
    #    shuffle(dts)
    mutual_information_matrix = np.empty(shape=(n_timesteps, n_timesteps), dtype=float)
    # have to broadcast this because shuffling on each process will have different results..
    #dts = system.comm.bcast(dts, root=0)
    #if system.rank != 0:
    #    slices = slice_tasks_for_parallel_workers(n_tasks=len(dts), n_workers=system.size-1)
    #    start = slices[system.rank-1].start
    #    stop = slices[system.rank-1].stop
    #    local_blocksize = stop - start
    with timeit() as calculation_block:
        for dt in ProgressBar(dts, position=system.rank,
                             desc=f'rank {system.rank} calculating mutual information matrix.'):
            current_dt = dt
            print(current_dt)
            if current_dt != 0:
                n_dt = int(current_dt / system.dt)
                len_current_diagonal = int(len_main_diagonal - n_dt)
                Q_for_this_dt = expm(system.generator_matrix*current_dt)
                # buffers to send to root
                lower_diagonal = np.empty(shape=len_current_diagonal, dtype=float)
                upper_diagonal = np.empty(shape=len_current_diagonal, dtype=float)
                # starting point for every diagonal, will increment each index by 1
                i = n_dt
                j = 0
                print(f'{n_dt=}')
                for k in range(len_current_diagonal):
                    print(f'i={i}, j={j}')
                    Probability_distribution_at_t_i = system.results[i]
                    Probability_distribution_at_t_j = system.results[j]
                    # initialize the sums to 0
                    lower_diagonal_element = 0
                    upper_diagonal_element = 0
                    # I(X(t_i);Y(t_j)) sum iteration block
                    for x_point in x_map:
                        for y_point in y_map:
                            idx = x_map[x_point]
                            idy = y_map[y_point]
                            # THESE ARE PROBABILITIES AT DIFFERENT TIMEPOINTS
                            # LOWER TRIANGLE ELEMENTS
                            Px_i = np.sum(Probability_distribution_at_t_i[idx])
                            Py_j = np.sum(Probability_distribution_at_t_j[idy])
                            Q_tilde_lower = sum([Q_for_this_dt[j, i] for x in idx for y in idy])
                            Q_ids = [(j, i) for i in idx for j in idy]
                            #print(f'Q elements added: {Q_ids}\n')
                            lower_diagonal_term = Q_tilde_lower * Py_j * np.log(Q_tilde_lower / Px_i)
                            lower_diagonal_element += lower_diagonal_term
                            # UPPER TRIANGLE ELEMENTS
                            Px_j = np.sum(Probability_distribution_at_t_j[idx])
                            Py_i = np.sum(Probability_distribution_at_t_i[idy])
                            Q_tilde_upper = sum([Q_for_this_dt[i,j] for i in idx for j in idy])
                            Q_ids = [(j, i) for i in idx for j in idy]
                            #print(f'Q elements added: {Q_ids}\n\n')
                            upper_diagonal_term = Q_tilde_upper*Px_j*np.log(Q_tilde_upper/Py_i)
                            upper_diagonal_element += upper_diagonal_term

                    lower_diagonal[k] = lower_diagonal_element
                    upper_diagonal[k] = upper_diagonal_element
                    print(f'{lower_diagonal_element=}')
                    print(f'{upper_diagonal_element=}')
                    # increment the indices and go to next element in the diagonal
                    i += 1
                    j += 1

                np.fill_diagonal(mutual_information_matrix[n_dt:, :-n_dt], lower_diagonal)
                print(mutual_information_matrix[n_dt:, :-n_dt])
                np.fill_diagonal(mutual_information_matrix[:-n_dt, n_dt:], upper_diagonal)



            else:
                main_diagonal = np.empty(shape=len_main_diagonal, dtype=float)

                for i, ts in enumerate(ProgressBar(range(len_main_diagonal), position=system.rank,
                                                   desc=f'rank {system.rank} calculating mutual information.')):
                    probability_vector = system.results[ts]
                    # initialize the sum to 0
                    mutual_information = 0
                    # iterate through every term in the sum and add to the total
                    for x_point in x_map:
                        for y_point in y_map:
                            xy_point = x_point + y_point
                            if xy_point not in xy_map:
                                # skip cases where the concatenated coordinate tuples
                                # were never in the joint distribution to begin with
                                continue
                            idx = x_map[x_point]
                            idy = y_map[y_point]
                            idxy = xy_map[xy_point]
                            p_xy = np.sum(probability_vector[idxy])
                            if p_xy == 0:
                                # add zero to the sum if p_xy is 0
                                # need to do this because 0*np.log(0) returns an error
                                continue
                            p_x = np.sum(probability_vector[idx])
                            p_y = np.sum(probability_vector[idy])
                            this_term = p_xy * np.log(p_xy / (p_x * p_y))
                            # add any nonzero value to the sum
                            mutual_information += this_term
                    main_diagonal[i] = mutual_information

            np.fill_diagonal(mutual_information_matrix, main_diagonal)

    return mutual_information_matrix

def _get_point_mappings(system: MasterEquation, molecules: List[str]) ->  Dict[tuple, List[int]]:
    """Maps the indices of the microstates in the `system.constitutive_states` vector to their
    marginal probability distribution domain points for the specified molecular species.

    e.g. find where each (x,y,z) point in p(x,y,z) maps in p(x) if the set of (x,y,z) points were
    flattened into a 1-d array.

    """

    # indices that correspond to the selected molecular species in the ordered species list
    ids = [system.species.index(i) for i in sorted(molecules)]
    truncated_points = np.array(system.constitutive_states)[:, ids]

    # keys are tuple coordinates of the marginal distrubtion
    # values are the indices of the joint distribution points that map to the marginal point
    point_maps: Dict[tuple, List[int]] = {}

    curr_state_id = 0
    states_accounted_for: List[int] = []
    # begin mapping points until every state in the microstate space is accounted for
    while len(states_accounted_for) < len(truncated_points):
        # find the indices of degenerate domain points starting with the first state
        # and skipping iterations if that point has been accounted for
        if curr_state_id in states_accounted_for:
            curr_state_id += 1
            continue
        curr_state = truncated_points[curr_state_id]
        degenerate_ids = np.argwhere(
            np.all(truncated_points == curr_state, axis=1)).transpose().flatten().tolist()
        point_maps[tuple(curr_state)] = degenerate_ids
        # keep track of which states we have accounted for
        states_accounted_for += degenerate_ids
        curr_state_id += 1

    return point_maps


def _calc_mutual_information_OLD_VERSION(system: MasterEquation,
                                         X: List[str], Y: List[str],
                                         timestep: Union[str, int] = 'all') -> np.ndarray:
    """Old, much slower version of `calc_mutual_information`."""

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
            for ts in ProgressBar(range(n_timesteps), desc='calculating mutual information'):
                P = system.results[ts]
                mutual_information = 0
                for x in X_set:
                    for y in Y_set:
                        term = _single_term(x, y)
                        mutual_information += term
                mutual_information_array[ts] = mutual_information

    system.timings['t_calculate_mutual_information'] = calculation_time.elapsed

    return mutual_information_array
