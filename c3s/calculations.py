import numpy as np
from scipy.sparse.linalg import expm
from .utils import timeit, ProgressBar, slice_tasks_for_parallel_workers
from typing import List, Dict, Union
from mpi4py import MPI


class CalculationsMixin:

    def calculate_mutual_information(self, X: List[str], Y: List[str],
                                     timestep: Union[str, int] = 'all') -> np.ndarray:
        """Calculates the mutual information."""

        if timestep != 'all':
            raise ValueError("other timestep values not implemented yet..")
        try:
            n_timesteps = len(self.results)
        except TypeError:
            raise ValueError('No data found in self.results attribute.')

        x_map = self._get_point_mappings(X)
        y_map = self._get_point_mappings(Y)
        xy_map = self._get_point_mappings(X + Y)

        if self.parallel:
            slices = slice_tasks_for_parallel_workers(n_tasks=n_timesteps, n_workers=self.size)
            sendcounts = tuple(slices[i].stop - slices[i].start for i in range(self.size))
            displacements = tuple(slice_.start for slice_ in slices)
            start = slices[self.rank].start
            stop = slices[self.rank].stop
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
            for i, ts in enumerate(ProgressBar(range(start,stop), position=self.rank,
                                               desc=f'rank {self.rank} calculating mutual information.')):
                probability_vector = self.results[ts]
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

            if self.parallel:
                self.comm.Allgatherv(sendbuf=mutual_information_local,
                                       recvbuf=(mutual_information_global, sendcounts, displacements, MPI.DOUBLE))

        self.timings['t_calculate_mutual_information'] = calculation_block.elapsed

        return mutual_information_global

    def calculate_marginal_probability_evolution(self, molecules: List[str]):
        """"""
        if self.results is None:
            raise ValueError('No data found in self.results attribute.')
        P = self.results
        point_maps = self._get_point_mappings(molecules)
        distribution: Dict[tuple, np.ndarray] = {}
        for point, map in point_maps.items():
            distribution[point] = np.array([sum(vec[map]) for vec in P])

        return distribution

    def _get_point_mappings(self, molecules: List[str]) -> Dict[tuple, List[int]]:
        """Maps the indices of the microstates in the `system.constitutive_states` vector to their
        marginal probability distribution domain points for the specified molecular species.

        e.g. find where each (x,y,z) point in p(x,y,z) maps in p(x) if the set of (x,y,z) points were
        flattened into a 1-d array.

        """

        # indices that correspond to the selected molecular species in the ordered species list
        ids = [self.species.index(i) for i in sorted(molecules)]
        truncated_points = np.array(self.constitutive_states)[:, ids]

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

    def calculate_mutual_information_matrix(self, X: List[str], Y: List[str]) -> np.ndarray:
        """Calculates the mutual information between every pair of timepoints."""

        n_timesteps = len(self.results)
        dts = [self._dt*timestep for timestep in range(n_timesteps)]
        dts_matrix = np.zeros(shape=(len(dts), len(dts)))
        for i, dt in enumerate(dts):
            np.fill_diagonal(dts_matrix[:-i, i:], dt)  # upper triangle
            np.fill_diagonal(dts_matrix[i:, :-i], dt)  # lower triangle

        MI_matrix = np.empty(shape=(len(dts), len(dts)), dtype=float)

        x_map = self._get_point_mappings(X)
        y_map = self._get_point_mappings(Y)
        xy_map = self._get_point_mappings(X + Y)

        Q_dict = {}
        for dt in dts:
            Q_dict[dt] = expm(self.generator_matrix * dt)

        def calc_matrix_element(i, j):
            #print(f'\n\ndoing i = {i}, j = {j}')
            # p(n_A, n_B, n_C) at t=i
            P_i = self.results[i]
            # p(n_A, n_B, n_C) at t=j
            P_j = self.results[j]
            dt = dts_matrix[i, j]
            Q = Q_dict[dt]

            MI = 0
            for x_indices in x_map.values():
                for y_indices in y_map.values():
                    p_x = np.sum(P_i[x_indices])
                    p_y = np.sum(P_j[y_indices])
                    if i > j:
                        Qtilde = np.sum([Q[i, j] * P_j[j] for i in x_indices for j in y_indices])
                    elif i < j:
                        Qtilde = np.sum([Q[j, i] * P_i[i] for i in x_indices for j in y_indices])
                    if Qtilde == 0:
                        term = 0
                    else:
                        term = Qtilde * np.log(Qtilde / (p_x * p_y))
                    MI += term
                    #print(f'x point: {x_indices}\ny point: {y_indices}\nterm = {term}\n')
            return MI

        with timeit() as calculation_block:

            for i in ProgressBar(range(len(MI_matrix))):
                for j in range(len(MI_matrix)):
                    if i != j:
                        MI_matrix[i, j] = calc_matrix_element(i, j)

            main_diagonal = self.calculate_mutual_information(X, Y)
            np.fill_diagonal(MI_matrix, main_diagonal)

        return MI_matrix


def _calc_mutual_information_OLD_VERSION(system,
                                         X: List[str], Y: List[str],
                                         timestep: Union[str, int] = 'all') -> np.ndarray:
    """Old, much slower version of `calc_mutual_information`. Only keeping here for testing/double checking
    calculations."""

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
