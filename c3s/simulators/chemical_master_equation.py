from typing import Dict
from collections import namedtuple
from pathlib import Path
import numpy as np
from .reaction_network import ReactionNetwork
from ..utils import timeit
from ..sparse_matrix import SparseMatrix
from ..h5io import CMEWriter
from ..marginalization import get_marginalized_trajectory, get_average_copy_number
from ..mutual_information import calculate_mutual_information
from ..math_utils import (generate_subspace, combine_state_spaces, vector_to_number, binary_search, calculate_Omega,
                          calculate_B, calculate_Q, calculate_imu_timestep, calculate_expm_timestep)


class ChemicalMasterEquation:
    """Simulator class of the Chemical Master Equation (CME).

    Uses the Chemical Master Equation to numerically integrate the time
    evolution of the probability trajectory of a chemical system.

    The only input required by the user is the config file that specifies the elementary chemical reactions
    and kinetic rates, and the initial nonzero population numbers of each species. The constructor will
    build the full constitutive state space in `self.constitutive_states` and the generator matrix in
    `self.G`. To run the simulator, call the `self.run()` method with start, stop, and dt
    arguments. The full P(t) trajectory will be stored in `self.trajectory`.

    """

    np.seterr(under='raise', over='raise')

    # use a namedtuple to store trajectory data/metadata cleanly
    CMETrajectory = namedtuple("CMETrajectory", ['trajectory', 'method', 'dt', 'rates', 'Omega', 'B', 'Q'])

    def __init__(self, config=None, initial_copy_numbers=None, empty=False):
        """

        Parameters
        ----------
        config : str or Dict
            path to yaml config file that specifies chemical reactions and elementary rates
        initial_copy_numbers : Dict[str: int] (default=None)
            dictionary of initial species copy numbers,
            if a species population is not specified, it's initial population is taken to be 0
        empty : bool (default=False)
            set to `True` to build system by reading data file

        """

        self.reaction_network = ReactionNetwork(config)
        self.species = self.reaction_network.species
        self._rates = self.reaction_network.rates
        if initial_copy_numbers is not None:
            for species in initial_copy_numbers.keys():
                if species not in self.species:
                    raise KeyError(f'{species} is not a valid species for this system')
        self._initial_copy_numbers = initial_copy_numbers
        self._empty = empty
        # dictionary to hold timings of various codeblocks for benchmarking
        self.timings: Dict[str, float] = {}
        self._constitutive_states = self._set_constitutive_states()
        self._base = self._constitutive_states.max() + 1 if not empty else None
        self._states_as_numbers = vector_to_number(self._constitutive_states, self._base) if not empty else None
        self._generator_matrix = self._set_generator_matrix()
        self._initial_state = np.array([
            self._initial_copy_numbers[species] if species in self._initial_copy_numbers
            else 0 for species in self.species]) if initial_copy_numbers else None
        self._initial_state_index = binary_search(
            self._states_as_numbers, vector_to_number(self._initial_state, self._base)) if self._initial_state is not None else 0
        # will be filled with `self.run()` method is called
        self._Trajectory = None

    def _set_constitutive_states(self):
        if self._empty:
            return None
        with timeit() as t_build_state_space:
            constitutive_states = self._build_state_space()
        self.timings['t_build_state_space'] = t_build_state_space.elapsed
        return constitutive_states

    def _set_generator_matrix(self):
        if self._empty:
            return None
        with timeit() as t_set_generator_matrix:
            generator_matrix = self._build_generator_matrix()
        self.timings['t_build_generator_matrix'] = t_set_generator_matrix.elapsed
        return generator_matrix

    def _build_state_space(self):
        subspaces = [generate_subspace(Constraint) for Constraint in self.reaction_network.constraints]
        if len(subspaces) > 1:
            return combine_state_spaces(*subspaces)
        else:
            return np.stack(subspaces[0])

    def _build_generator_matrix(self):
        M, K, N, base = self.M, self.K, self.N, self._base

        G_rows = [i for i in range(M)]
        G_cols = [j for j in range(M)]
        G_values = [0.0 for _ in range(M)]
        reactions_as_numbers = vector_to_number(self.reaction_network.reaction_matrix, base)
        for j, state_j in enumerate(self._states_as_numbers):
            for k, reaction in enumerate(reactions_as_numbers):
                state_i = state_j + reaction
                i = binary_search(self._states_as_numbers, state_i)
                if i == -1:
                    continue
                ids = self.reaction_network.species_in_reaction[k]
                rate = self._rates[k]
                h = np.prod([self.states[j,n] for n in ids])
                # overall reaction propensity for j -> i
                propensity = h * rate
                if propensity != 0:
                    G_rows.append(i)
                    G_cols.append(j)
                    G_values.append(propensity)
                    G_values[j] -= propensity

        return SparseMatrix(np.array(G_rows), np.array(G_cols), np.array(G_values))

    def run(self, method='IMU', dt=1.0, N_timesteps=None,
            to_steady_state=False, overwrite=False, is_continued_run=False):
        """runs the chemical master equation simulation

        Parameters
        ----------
        method : 'IMU' or 'EXPM' (default='IMU')
            ...
        dt : float (default=1.0)
            value of timestep that multiplies into generator matrix
        N_timesteps : int (default=None)
            number of simulation timesteps, must be specified if `to_steady_state=False`
        to_steady_state : bool (default=False)
            set to `True` to run simulation until steady state condition has been achieved
        overwrite : bool (default=False)
            set to `True` to rerun a simulation from scratch and overwrite data in `self.Trajectory`
        is_continued_run : bool (default=False)
            set to `True` to run continued simulation and concatenate results with `self.Trajectory`
        """
        # handle errors early
        if (N_timesteps is None) and (not to_steady_state):
            raise ValueError("`N_timesteps` must be specified if `to_steady_state=False`.")
        if (self._Trajectory is not None) and (not overwrite) and (not is_continued_run):
            raise AttributeError("Data already found in `self.Trajectory`. To overwrite this data, set `overwrite=True`.")

        if method == 'IMU':
            with timeit() as t_build_B_matrix:
                Omega = calculate_Omega(self.G)
                B = calculate_B(self.G, self.M, Omega)
            self.timings['t_build_B_matrix'] = t_build_B_matrix.elapsed
            function = calculate_imu_timestep
            args = [Omega*dt, B]
            Q = None
        elif method == 'EXPM':
            with timeit() as t_matrix_exponential:
                Q = calculate_Q(self.G, dt)
            self.timings['t_matrix_exponential'] = t_matrix_exponential.elapsed
            function = calculate_expm_timestep
            args = [Q]
            Omega, B = None, None
        else:
            raise ValueError(f"`{method}` must be 'IMU' or 'EXPM'")

        if to_steady_state:
            trajectory = self._run_until_steady_state(method, function, args, is_continued_run)
        else:
            trajectory = self._run_N_timesteps(N_timesteps, method, function, args, is_continued_run)

        self._Trajectory = self.CMETrajectory(trajectory=trajectory,
                                              method=method,
                                              dt=dt,
                                              rates=self._rates,
                                              Omega=Omega,
                                              B=B,
                                              Q=Q)

    def _run_until_steady_state(self, method, function, args, is_continued_run):
        if is_continued_run:
            trajectory = [self.trajectory[-1]]
        else:
            trajectory = [np.zeros(self.M)]
            trajectory[0][self._initial_state_index] = 1.0
        with timeit() as t_run:
            ts = 0
            while True:
                trajectory.append(function(trajectory[ts], *args))
                P_difference = trajectory[ts + 1] - trajectory[ts]
                if P_difference.max() < 1e-6:
                    break
                ts += 1
        self.timings[f't_run_{method}'] = t_run.elapsed
        return np.vstack((self.trajectory, trajectory)) if is_continued_run else np.asarray(trajectory)

    def _run_N_timesteps(self, N_timesteps, method, function, args, is_continued_run):
        if is_continued_run:
            trajectory = np.vstack([self.trajectory, np.empty((N_timesteps, self.M))])
            start = len(self.trajectory) - 1
            stop = start + N_timesteps
        else:
            trajectory = np.empty((N_timesteps, self.M))
            trajectory[0] = np.zeros(self.M)
            trajectory[0, self._initial_state_index] = 1.0
            start, stop = 0, N_timesteps - 1
        with timeit() as t_run:
            for ts in range(start, stop):
                trajectory[ts + 1] = function(trajectory[ts], *args)
        self.timings[f't_run_{method}'] = t_run.elapsed
        return trajectory

    def update_rates(self, new_rates):
        """updates `self.rates` and `self.G` with new transition rates"""
        for rate_name, new_rate in new_rates.items():
            for k, reaction in enumerate(self.reaction_network.reactions):
                if reaction.rate_name == rate_name:
                    updated_reaction = reaction._replace(rate=new_rate)
                    self.reaction_network.reactions[k] = updated_reaction
                    self._rates[k] = new_rate
                    break
            else:
                raise KeyError(f'{rate_name} is not a valid rate for this system. Valid rates'
                               f'are listed in `self.rates`.')
        # recreate G with new rates
        self._generator_matrix = self._build_generator_matrix()

    def write_system(self, filename, mode='x'):
        """writes system data to HDF5 file"""
        with CMEWriter(filename=filename, mode=mode, system=self) as Writer:
            Writer._write_system()

    def write_trajectory(self, filename, mode='r+', trajectory_name=None):
        """write trajectory data to HDF5 file"""
        if self._Trajectory is None:
            raise AttributeError('no data in `system.Trajectory`.')
        # if the file does not exist yet
        if not Path(filename).is_file():
            self.write_system(filename)
        with CMEWriter(filename, mode, system=self) as Writer:
            Writer._write_trajectory(trajectory_name=trajectory_name)

    def write_mutual_information(self, filename, trajectory_name, data, X, Y, base=2, mode='r+'):
        """"""
        X = [X] if isinstance(X, str) else X
        Y = [Y] if isinstance(Y, str) else Y
        with CMEWriter(filename, mode, system=self) as Writer:
            Writer._write_mutual_information(trajectory_name, data, base, X, Y)

    def write_avg_copy_number(self, filename, trajectory_name, species, data, mode='r+'):
        with CMEWriter(filename, mode, system=self) as Writer:
            Writer._write_avg_copy_number(trajectory_name, species, data)

    def write_marginalized_trajectory(self, filename, trajectory_name, species_subset, marginalized_trajectory, mode='r+'):
        with CMEWriter(filename, mode, system=self) as Writer:
            Writer._write_marginalized_trajectory(trajectory_name, species_subset, marginalized_trajectory)

    def calculate_mutual_information(self, X, Y, base=2):
        X = [X] if isinstance(X, str) else X
        Y = [Y] if isinstance(Y, str) else Y
        with timeit() as t_calculate_mi:
            mi = calculate_mutual_information(X, Y, system=self, base=base)
        self.timings['t_calculate_mi'] = t_calculate_mi.elapsed
        return mi

    def calculate_marginalized_trajectory(self, species_subset):
        return get_marginalized_trajectory(species_subset, system=self)

    def calculate_average_copy_number(self, species):
        return get_average_copy_number(species, system=self)

    @property
    def N(self) -> int:
        """N is the total number of unique chemical species in the reaction network"""
        return len(self.reaction_network.species)
    @property
    def K(self) -> int:
        """K is the total number of reactions in the reaction network"""
        return len(self.reaction_network.reaction_matrix)
    @property
    def M(self) -> int:
        """M is the total number of states in the state space"""
        return len(self._constitutive_states)
    @M.setter
    def M(self, value):
        self.M = value
    @property
    def G(self) -> SparseMatrix:
        """G is the MxM transition rate matrix in sparse matrix format"""
        return self._generator_matrix
    @G.setter
    def G(self, value):
        self._generator_matrix = value
    @property
    def states(self) -> np.ndarray:
        """states is MxN array where states[m,n] gives the copy number of the n'th species in the m'th state"""
        return self._constitutive_states
    @states.setter
    def states(self, value):
        self._constitutive_states = value
    @property
    def states_as_numbers(self) -> np.ndarray:
        """state vectors represented as numbers"""
        return self._states_as_numbers
    @property
    def Trajectory(self) -> namedtuple:
        """probability trajectory is contained in `Trajectory.trajectory`"""
        return self._Trajectory
    @Trajectory.setter
    def Trajectory(self, value):
        self._Trajectory = value
    @property
    def trajectory(self):
        return self._Trajectory.trajectory
