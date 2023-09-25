from typing import List, Dict
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm

from .reaction_network import ReactionNetwork
from ..utils import timeit
from ..sparse_matrix import SparseMatrix
from ..h5io import CMEWriter
from ..marginalization import marginalize_trajectory, average_copy_number
from ..mutual_information import mutual_information
from ..math_utils import (generate_subspace,
                          combine_state_spaces,
                          vector_to_number,
                          binary_search,
                          IMU_timestep)


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

    def __init__(self, config=None, initial_populations=None, empty=False):
        """

        Parameters
        ----------
        config : str or Dict
            path to yaml config file that specifies chemical reactions and elementary rates
        initial_populations : Dict[str: int]
            dictionary of initial species populations,
            if a species population is not specified, it's initial population is taken to be 0
        empty : bool (default=False)
            set to `True` to build system by reading data file

        """

        # parse config file information
        self.reaction_network = ReactionNetwork(config)
        self.species = self.reaction_network.species
        self._rates = self.reaction_network.rates
        self._initial_populations = initial_populations

        if not empty:
            # dictionary to hold timings of various codeblocks for benchmarking
            self.timings: Dict[str, float] = {}
            # build system from config
            self._constitutive_states = self._set_constitutive_states()
            self._base = self._constitutive_states.max() + 1
            self._states_as_numbers = self._convert_states_to_numbers(N=self.N, base=self._base)
            self._generator_matrix = self._set_generator_matrix()
            self._initial_state, self._initial_state_index = self._determine_initial_state()
        else:
            self._constitutive_states = None
            self._states_as_numbers = None
            self._generator_matrix = None
            self._initial_state = None
            self._initial_state_index = 0

        # will be filled with `self.run()` method is called
        self._Trajectory = None

    # use nice Pythonic attribute access
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
        """probability trajectory is contained in Trajectory.trajectory"""
        return self._Trajectory
    @Trajectory.setter
    def Trajectory(self, value):
        self._Trajectory = value

    def _set_constitutive_states(self):
        with timeit() as t_build_state_space:
            constitutive_states = self._build_state_space()
        self.timings['t_build_state_space'] = t_build_state_space.elapsed
        return constitutive_states

    def _build_state_space(self):
        subspaces = [generate_subspace(Constraint) for Constraint in self.reaction_network.constraints]
        if len(subspaces) > 1:
            return combine_state_spaces(*subspaces)
        else:
            return np.stack(subspaces[0])

    def _convert_states_to_numbers(self, N, base):
        return vector_to_number(self._constitutive_states, N, base)

    def _determine_initial_state(self):

        if self._initial_populations is None:
            return None, 0

        # ensure all species are valid
        for species in self._initial_populations.keys():
            if species not in self.species:
                raise KeyError(f'{species} is not a valid species. It must be in one of the '
                               f'chemical reactions specified in the original config file.')

        initial_state_as_vector = np.array([self._initial_populations[species]
                                            if species in self._initial_populations else 0
                                            for species in self.species])
        initial_state_as_number = vector_to_number(initial_state_as_vector, self.N, self._base)
        initial_state_index = binary_search(self._states_as_numbers, initial_state_as_number)

        return initial_state_as_vector, initial_state_index

    def _set_generator_matrix(self):
        with timeit() as t_set_generator_matrix:
            generator_matrix = self._build_generator_matrix()
        self.timings['t_build_generator_matrix'] = t_set_generator_matrix.elapsed
        return generator_matrix

    def _build_generator_matrix(self):
        M, K, N, base = self.M, self.K, self.N, self._base
        # gives which elements of G the k'th reaction is responsible for
        #self._k_to_G_map = {k: [] for k in range(self.K)}

        G_rows = [i for i in range(M)]
        G_cols = [j for j in range(M)]
        G_values = [0.0 for _ in range(M)]
        reactions_as_numbers = vector_to_number(self.reaction_network.reaction_matrix, N, base)
        for j, state_j in enumerate(self._states_as_numbers):
            for k, reaction in enumerate(reactions_as_numbers):
                state_i = state_j + reaction
                # returns index of state_i if exists, else -1
                i = binary_search(self._states_as_numbers, state_i)
                if i == -1:
                    # state_j + reaction_k was not in state space
                    continue
                # indices of which species are involved in k'th reaction
                ids = self.reaction_network.species_in_reaction[k]
                # the elementary transition rate of the k'th reaction
                rate = self._rates[k]
                # the combinatorial factor for the k'th reaction firing in the j'th state
                h = np.prod([self.states[j,n] for n in ids])
                # overall reaction propensity for j -> i
                propensity = h * rate
                if propensity != 0:
                    G_rows.append(i)
                    G_cols.append(j)
                    G_values.append(propensity)
                    G_values[j] -= propensity
                    #self._k_to_G_map[k].append((i,j))

        return SparseMatrix(np.array(G_rows), np.array(G_cols), np.array(G_values))

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

    def reset_rates(self):
        """resets `self.rates` to the values of the original config file"""
        rates_from_config = [rate for rate in self.reaction_network._original_config['reactions'].values()]
        original_rates = {rate[0]: rate[1] for rate in rates_from_config}
        self.update_rates(original_rates)

    def run(self, dt, method, N_timesteps=None, to_steady_state=False, overwrite=False, continued=False):
        """runs the chemical master equation simulation

        Parameters
        ----------
        dt : float
            value of timestep that multiplies into generator matrix
        method : 'IMU' or 'EXPM'
            whether to call `self._run_IMU()` or `self._run_EXPM()`
        N_timesteps : int (default=None)
            number of simulation timesteps, must be specified if `to_steady_state=False`
        to_steady_state : bool (default=False)
            set to `True` to run simulation until steady state condition has been achieved
        overwrite : bool (default=False)
            set to `True` to rerun a simulation from scratch and overwite data in `self.Trajectory`
        continued : bool (default=False)
            set to `True` to run continued simulation and concatenate results with `self.Trajectory`

        """

        # handle errors early
        if (N_timesteps is None) and (not to_steady_state):
            raise ValueError("`N_timesteps` must be specified if `to_steady_state=False`.")
        if (self._Trajectory is not None) and (not overwrite):
            # ok if we are concatenating new data
            if not continued:
                raise AttributeError(
                    "Data already found in `self.Trajectory`. To overwrite this data, set `overwrite=True`.")

        if method == 'IMU':
            self._run_IMU(N_timesteps, dt, continued)
        if method == 'EXPM':
            self._run_EXPM(N_timesteps, dt, continued)

    def _run_IMU(self, N_timesteps, dt, continued, scale_factor=1.1):
        """inverse marginalized uniformization method"""

        with timeit() as t_build_B_matrix:
            Omega = abs(max(self.G.values, key=abs)) * scale_factor
            OmegaT = Omega * dt
            B = self._build_B_matrix(Omega)
        self.timings['t_build_B_matrix'] = t_build_B_matrix.elapsed

        trajectory = np.empty(shape=(N_timesteps, self.M), dtype=np.float64)
        if continued:
            trajectory[0] = IMU_timestep(p_0=self._Trajectory.trajectory[-1], B=B, OmegaT=OmegaT)
        else:
            trajectory[0] = np.zeros(shape=self.M, dtype=np.float64)
            # fixing initial probability to be 1 in the initial state
            i = self._initial_state_index
            trajectory[0,i] = 1.0

        with timeit() as t_run_IMU:
            for ts in range(N_timesteps - 1):
                trajectory[ts + 1] = IMU_timestep(p_0=trajectory[ts], B=B, OmegaT=OmegaT)
        self.timings['t_run_IMU'] = t_run_IMU.elapsed

        if continued:
            trajectory = np.vstack([self._Trajectory.trajectory, trajectory])

        self._Trajectory = self.CMETrajectory(
            trajectory=trajectory, method='IMU', dt=dt, rates=self._rates, Omega=Omega, B=B, Q=None)

    def _build_B_matrix(self, Omega):
        B_values = self.G.values / Omega
        B_values[:self.M] = B_values[:self.M] + 1
        return SparseMatrix(self.G.rows, self.G.columns, B_values)

    def _run_EXPM(self, N_timesteps, dt, continued):
        """scipy sparse matrix exponentiation method"""

        with timeit() as t_matrix_exponential:
            Q = expm(self._generator_matrix.to_dense() * dt)
        self.timings['t_matrix_exponential'] = t_matrix_exponential.elapsed

        trajectory = np.empty(shape=(N_timesteps, self.M), dtype=np.float64)
        if continued:
            trajectory[0] = Q.dot(self._Trajectory.trajectory[-1])
        else:
            trajectory[0] = np.zeros(shape=self.M, dtype=np.float64)
            # fixing initial probability to be 1 in the initial state
            i = self._initial_state_index
            trajectory[0,i] = 1.0

        with timeit() as t_run_EXPM:
            for ts in range(N_timesteps - 1):
                trajectory[ts + 1] = Q.dot(trajectory[ts])
        self.timings['t_run_EXPM'] = t_run_EXPM.elapsed

        if continued:
            trajectory = np.vstack([self._Trajectory.trajectory, trajectory])

        self._Trajectory = self.CMETrajectory(
            trajectory=trajectory, method='EXPM', dt=dt, rates=self._rates, Omega=None, B=None, Q=Q)

    def write_system(self, filename, mode='x'):
        """writes system data to HDF5 file"""
        with CMEWriter(filename=filename, mode=mode, system=self) as Writer:
            Writer._write_system()

    def write_trajectory(self, filename, mode='r+', trajectory_name=None):
        """write trajectory data to HDF5 file"""

        self._check_if_Trajectory_exists()
        # if the file does not exist yet
        if not Path(filename).is_file():
            self.write_system(filename=filename)

        with CMEWriter(filename=filename, mode=mode, system=self) as Writer:
            Writer._write_trajectory(trajectory_name=trajectory_name)

    def write_calculation(self, filename, mode='r+'):
        ...

    def calculate_mutual_information(self, X, Y, base=2):
        self._check_if_Trajectory_exists()
        return mutual_information(system=self, X=X, Y=Y, base=base)

    def calculate_marginalized_trajectory(self, species_subset):
        self._check_if_Trajectory_exists()
        return marginalize_trajectory(system=self, species_subset=species_subset)

    def calculate_average_copy_number(self, species):
        self._check_if_Trajectory_exists()
        return average_copy_number(system=self, species=species)

    def _check_if_Trajectory_exists(self):
        if self._Trajectory is None:
            raise AttributeError('no data in `self.Trajectory`.')
