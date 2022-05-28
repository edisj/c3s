import numpy as np
from .simulators import MasterEquation, MPI_ON
from .utils import timeit, ProgressBar, slice_tasks_for_parallel_workers
from typing import List, Dict, Union


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

    if MPI_ON:
        slices = slice_tasks_for_parallel_workers(n_tasks=n_timesteps, n_workers=system.size)
        sendcounts = n_timesteps * np.array([slices[i].stop - slices[i].start for i in range(system.size)])
        start = slices[system.rank].start
        stop = slices[system.rank].stop
        local_blocksize = stop - start
        # global buffer that all processes will combine their data into
        mutual_information_global = np.empty(shape=n_timesteps, dtype=float)
        # local buffer that only this process sees
        mutual_information_local = np.empty(shape=local_blocksize, dtype=float)
    else:
        start = 0
        stop = n_timesteps
        local_blocksize = stop - start
        mutual_information_local = mutual_information_global = np.empty(shape=local_blocksize, dtype=float)

    with timeit() as calculation_block:
        for i, ts in enumerate(ProgressBar(range(start,stop),
                                           desc=f'rank {system.rank} calculating mutual information')):
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

        if MPI_ON:
            system.comm.Gatherv(sendbuf=mutual_information_local,
                                recvbuf=(mutual_information_global, sendcounts),
                                root=0)

    system.timings['t_get_point_mappings'] = get_point_mappings_block.elapsed
    system.timings['t_calculate_mutual_information'] = calculation_block.elapsed

    return mutual_information_global


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
