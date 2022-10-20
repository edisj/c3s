import math
import numpy as np
import h5py
from collections import namedtuple
from typing import List, Dict
from scipy.sparse.linalg import expm
from .utils import split_tasks_for_workers, ProgressBar, timeit


class CalculationsMixin:
    _system_file: h5py.File
    timings: dict
    _rates: List
    _system_name: str
    _run_name: str
    _constitutive_states: np.ndarray
    _species: List[str]
    _dt: float
    G: np.ndarray
    P_trajectory: np.ndarray

    def calculate_instantaneous_mutual_information(self, X, Y, version='diagonal', timestep='all', base=2):

        calculate_mi_dispatch = {'diagonal': self._calculate_mutual_information_diagonal,
                                 'matrix': self._calculate_mutual_information_matrix}

        if timestep != 'all':
            raise ValueError("other timestep values not implemented yet..")
        try:
            N_timesteps = len(self.P_trajectory)
        except TypeError:
            raise ValueError('No data found in `self.P_trajectory`.')
        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            Y = [Y]
        X = sorted(X)
        Y = sorted(Y)
        if X + Y != sorted(X + Y):
            X, Y = Y, X

        DeltaTuple = namedtuple("Deltas", "x y xy")
        Deltas = DeltaTuple(
            self._get_Delta_vectors(X),
            self._get_Delta_vectors(Y),
            self._get_Delta_vectors(X + Y))

        with timeit() as calculation_block:
            mutual_information = calculate_mi_dispatch[version](Deltas, base)
        self.timings[f't_calculate_mi_{version}'] = calculation_block.elapsed

        '''if self._system_file:
            group = self._system_file[self._run_name]
            mi_dataset = group.create_dataset('mutual_information',
                                               shape=mutual_information.shape,
                                               dtype=mutual_information.dtype)
            mi_dataset.attrs['rates'] = [rate[1] for rate in self._rates]
            mi_dataset.attrs['dt'] = self._dt
            mi_dataset.attrs['logbase'] = base
            mi_dataset[:] = mutual_information'''

        self._mutual_information = mutual_information
        return mutual_information

    def _get_Delta_vectors(self, molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        # indices that correspond to the selected molecular species in the ordered species list
        ids = [self._species.index(n) for n in molecules]
        truncated_points = np.array(self._constitutive_states)[:, ids]

        Delta_vectors: Dict[tuple, np.ndarray] = {}

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
            Dv = np.all(truncated_points == curr_state, axis=1).astype(np.float64)
            Delta_vectors[tuple(curr_state)] = Dv
            # keep track of which states we have accounted for
            states_accounted_for += np.argwhere(Dv == 1).transpose().tolist()
            curr_state_id += 1

        return Delta_vectors

    def _calculate_mutual_information_diagonal(self, Deltas, base):
        """"""

        N_timesteps = len(self.P_trajectory)
        mut_inf = np.zeros(shape=N_timesteps, dtype=np.float64)

        for ts in ProgressBar(range(N_timesteps), desc=f'calculating mutual information...'):
            P = self.P_trajectory[ts]
            mi_sum = 0
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
                    mi_sum += this_term
            mut_inf[ts] = mi_sum

        '''if self.parallel:
            global_mi = np.zeros(shape=N_timesteps, dtype=np.float64)
            sendcounts = self.comm.allgather(blocksize)
            displacements = self.comm.allgather(start)
            self.comm.Allgatherv(sendbuf=local_mi, recvbuf=(global_mi, sendcounts, displacements, MPI.DOUBLE))'''

        return mut_inf

    def _calculate_mutual_information_matrix(self, Deltas, base):
        """calculates the pairwise instantaneous mutual information between
        every timepoint in the trajectory"""

        N_timesteps = len(self.P_trajectory)
        start, stop, blocksize = split_tasks_for_workers(N_tasks=N_timesteps, N_workers=self.size, rank=self.rank)

        tildeQ_indices = {
            (x, y): [(i, j) for i in np.argwhere(Dy == 1).flatten() for j in np.argwhere(Dx == 1).flatten()]
                     for x, Dx in Deltas.x.items() for y, Dy in Deltas.y.items()}
        tildeQ_Delta_matrices = {
            coordinates: self._make_tildeQ_Delta_matrix(indices) for coordinates, indices in tildeQ_indices.items()}

        global_mi_matrix = np.zeros(shape=(N_timesteps, N_timesteps), dtype=np.float64)
        local_dts = [self._dt * ts for ts in range(start, stop)]
        # using a generator expression to save on memory here
        local_Qs = (expm(self.G*dt) for dt in local_dts)
        for top_col_loc, Q in zip(range(start, stop), local_Qs):
            if top_col_loc == 0:
                diagonal = self._calculate_mutual_information_diagonal(Deltas, base)
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
                upper, lower = self._calculate_matrix_elements(i, j, Deltas, tildeQ_matrices, base)
                global_mi_matrix[i,j] = upper
                global_mi_matrix[j,i] = lower
                i += 1
                j += 1

        return global_mi_matrix

    def _make_tildeQ_Delta_matrix(self, indices):
        """helper function to construct tildeQ Delta matrices"""
        Delta_matrix = np.zeros(shape=self.G.shape, dtype=np.float64)
        for i,j in indices:
            Delta_matrix[i,j] += 1
        return Delta_matrix

    def _calculate_matrix_elements(self, i, j, Deltas, tildeQ_matrices, base):
        """calculates every matrix element for symmetric off diagonals"""

        # here I assume j is always at the later timepoint
        Pi = self.P_trajectory[i]
        Pj = self.P_trajectory[j]
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

    def calculate_marginal_probability_evolution(self, molecules):
        """"""

        if self.P_trajectory is None:
            raise ValueError('No data found in self.results attribute.')
        point_maps = self._get_point_mappings(molecules)
        distribution: Dict[tuple, np.ndarray] = {}
        for point, map in point_maps.items():
            distribution[point] = np.array([np.sum(self.P_trajectory[ts][map]) for ts in range(len(self.P_trajectory))])

        return distribution

    def calculate_average_population(self, species):
        """"""

        average_population = np.empty(shape=len(self.P_trajectory), dtype=np.float64)
        if self.P_trajectory is None:
            raise ValueError('No data found in self.results attribute.')
        P = self.P_trajectory
        point_maps = self._get_point_mappings(species)
        for ts in range(len(P)):
            total = 0
            for point, map in point_maps.items():
                assert len(point) == 1
                count = point[0]
                count_term = np.sum(count * P[ts][map])
                total += count_term
            average_population[ts] = total

        return average_population

    def _get_point_mappings(self, molecules):
        """Maps the indices of the microstates in the `system.states` vector to their
        marginal probability distribution domain points for the specified molecular species.

        e.g. find where each (x,y,z) point in p(x,y,z) maps in p(x) if the set of (x,y,z) points were
        flattened into a 1-d array.

        """

        if isinstance(molecules, str):
            molecules = [molecules]
        # indices that correspond to the selected molecular species in the ordered species list
        ids = [self._species.index(n) for n in sorted(molecules)]
        truncated_points = np.array(self._constitutive_states)[:, ids]

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

    def calculate_mutual_information(self, X, Y, timestep='all', base=2, run_name=None):
        """LEGACY VERSION

        Calculates the mutual information."""

        log = math.log2 if base == 2 else math.log

        if timestep != 'all':
            raise ValueError("other timestep values not implemented yet..")
        try:
            n_timesteps = len(self.P_trajectory)
        except TypeError:
            raise ValueError('No data found in `self.P_trajectory`.')

        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            Y = [Y]
        X = sorted(X)
        Y = sorted(Y)
        if X + Y != sorted(X + Y):
            X, Y = Y, X

        x_map = self._get_point_mappings(X)
        y_map = self._get_point_mappings(Y)
        xy_map = self._get_point_mappings(X + Y)

        start, stop, blocksize = split_tasks_for_workers(N_tasks=n_timesteps, N_workers=self.size, rank=self.rank)
        mutual_information_local = np.empty(shape=blocksize, dtype=np.float64)

        with timeit() as calculation_block:
            for i, ts in enumerate(ProgressBar(range(start, stop), position=self.rank,
                                               desc=f'rank {self.rank} calculating mutual information.')):
                P = self.P_trajectory[ts]
                mutual_information = 0
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
                        p_xy = np.sum(P[idxy])
                        if p_xy == 0:
                            # add zero to the sum if p_xy is 0
                            # need to do this because 0*np.log(0) returns an error
                            continue
                        p_x = np.sum(P[idx])
                        p_y = np.sum(P[idy])
                        this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
                        mutual_information += this_term
                mutual_information_local[i] = mutual_information
        self.timings['t_calculate_mutual_information'] = calculation_block.elapsed

        if self.parallel:
            mutual_information_global = np.empty(shape=n_timesteps, dtype=np.float64)
            sendcounts = self.comm.allgather(blocksize)
            displacements = self.comm.allgather(start)
            self.comm.Allgatherv(sendbuf=mutual_information_local,
                                 recvbuf=(mutual_information_global, sendcounts, displacements, MPI.DOUBLE))
        else:
            mutual_information_global = mutual_information_local
        if self._file:
            HDF5_group = self._file[self._system_name]
            run_name = 'run_' + str(len(self._file[self._system_name])) if run_name is None else run_name
            mi_dataset = HDF5_group.create_dataset(f'{run_name}/mutual_information', data=mutual_information_global)
            mi_dataset.attrs['rates'] = [rate[1] for rate in self._rates]
            mi_dataset.attrs['dt'] = self._dt

        return mutual_information_global
