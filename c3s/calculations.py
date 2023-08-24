import math
import numpy as np
import h5py
from collections import namedtuple
from typing import List, Dict
from scipy.sparse.linalg import expm
from .utils import timeit


class CalculationsMixin:
    _system_file: h5py.File
    timings: dict
    _rates: List
    _system_name: str
    _run_name: str
    _constitutive_states: np.ndarray
    species: List[str]
    _dt: float
    G: np.ndarray
    trajectory: np.ndarray

    def calculate_instantaneous_mutual_information(self, X, Y, base=2):

        try:
            N_timesteps = len(self.trajectory)
        except TypeError:
            raise ValueError('No data found in `self.trajectory`.')

        Deltas = self._fix_species_and_get_Deltas(X, Y)

        with timeit() as calculation_block:
            mut_inf = np.zeros(shape=N_timesteps, dtype=np.float64)
            mi_terms = [{} for _ in range(N_timesteps)]
            for ts in range(N_timesteps):
                P = self.trajectory[ts]
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
                        mi_terms[ts][xy] = this_term
                mut_inf[ts] = mi_sum

        self.timings[f't_calculate_insant_mi'] = calculation_block.elapsed
        self._mutual_information = mut_inf

        return mut_inf, mi_terms

    def _fix_species_and_get_Deltas(self, X, Y):
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

        return Deltas

    def _get_Delta_vectors(self, molecules):
        if isinstance(molecules, str):
            molecules = [molecules]
        # indices that correspond to the selected molecular species in the ordered species list
        ids = [self.species.index(n) for n in molecules]
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

        N_timesteps = len(self.trajectory)
        mut_inf = np.zeros(shape=N_timesteps, dtype=np.float64)

        mi_terms = {}
        for ts in range(N_timesteps):
            P = self.trajectory[ts]
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
                    mi_terms[xy] = this_term

            mut_inf[ts] = mi_sum

        return mut_inf, mi_terms

    def _calculate_mutual_information_matrix(self, Deltas, base):
        """calculates the pairwise instantaneous mutual information between
        every timepoint in the trajectory"""

        N_timesteps = len(self.trajectory)
        global_mi_matrix = np.zeros(shape=(N_timesteps, N_timesteps), dtype=np.float64)
        local_dts = [self._dt * ts for ts in range(N_timesteps)]
        # using a generator expression to save on memory here
        local_Qs = (expm(self.G*dt) for dt in local_dts)
        for top_col_loc, Q in zip(range(N_timesteps), local_Qs):
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

    def _calculate_matrix_elements(self, i, j, Deltas, tildeQ_matrices, base):
        """calculates every matrix element for symmetric off diagonals"""

        # here I assume j is always at the later timepoint
        Pi = self.trajectory[i]
        Pj = self.trajectory[j]

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

    def generate_analytic_MI_function(self, X, Y, base=2):

        Deltas = self._fix_species_and_get_Deltas(X, Y)

        indices_for_each_term = [[np.argwhere(Deltas.xy[x+y]).T.tolist()[0], np.argwhere(Dx).T.tolist()[0], np.argwhere(Dy).T.tolist()[0]]
                                for x,Dx in Deltas.x.items() for y,Dy in Deltas.y.items() if x+y in Deltas.xy]

        def analytic_function(P, base=base):
            term_values = []
            for ids in indices_for_each_term:
                Pxy = sum([P[i] for i in ids[0]])
                Px = sum([P[i] for i in ids[1]])
                Py = sum([P[i] for i in ids[2]])
                if Pxy == 0:
                    this_term = 0.0
                else:
                    this_term = Pxy * math.log( Pxy / (Px*Py), base)
                term_values.append(this_term)
            return sum(term_values)

        return analytic_function

    def _get_analytic_string(self, X, Y):

        Deltas = self._fix_species_and_get_Deltas(X, Y)

        indices_for_each_term = [[np.argwhere(Deltas.xy[x+y]).T.tolist()[0], np.argwhere(Dx).T.tolist()[0], np.argwhere(Dy).T.tolist()[0]]
                                for x,Dx in Deltas.x.items() for y,Dy in Deltas.y.items() if x+y in Deltas.xy]
        string_per_term = []
        for ids in indices_for_each_term:
            Pxy = [f"p{i + 1}" for i in ids[0]]
            Px = [f"p{i + 1}" for i in ids[1]]
            Py = [f"p{i + 1}" for i in ids[2]]

            term = f"{Pxy}log({Pxy}/({Px}{Py}))"
            string_per_term.append(term)

        return string_per_term

    def calculate_marginal_probability_evolution(self, molecules):
        """"""

        if self.trajectory is None:
            raise ValueError('No data found in self.trajectory.')
        point_maps = self._get_point_mappings(molecules)
        distribution: Dict[tuple, np.ndarray] = {}
        for point, map in point_maps.items():
            distribution[point] = np.array([np.sum(self.trajectory[ts][map]) for ts in range(len(self.trajectory))])

        return distribution

    def calculate_average_population(self, species):
        """"""

        average_population = np.empty(shape=len(self.trajectory), dtype=np.float64)
        if self.trajectory is None:
            raise ValueError('No data found in self.trajectory.')
        P = self.trajectory
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
        ids = [self.species.index(n) for n in sorted(molecules)]
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
