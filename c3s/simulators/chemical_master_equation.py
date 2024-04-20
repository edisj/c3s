from typing import Dict
from collections import namedtuple
import numpy as np
from .reaction_network import ReactionNetwork
from ..utils import time_this_code
from ..sparse_matrix import SparseMatrix
from ..h5io import Write
from ..calculations import Calculate
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
    CMETrajectory = namedtuple("CMETrajectory",
                               ['trajectory', 'method', 'dt', 'N_timesteps', 'rates', 'Omega', 'B', 'Q'])

    def __init__(self, config=None, initial_populations=None, empty=False):
        """

        Parameters
        ----------
        config : str or Dict
            path to yaml config file that specifies chemical reactions and elementary rates
        initial_populations : Dict[str: int] (default=None)
            dictionary of initial species copy numbers,
            if a species population is not specified, it's initial population is taken to be 0
        """

        self.reaction_network = ReactionNetwork(config)
        self.species = self.reaction_network.species
        self._rates = self.reaction_network.rates
        self.write = Write(system=self)
        self.calculate = Calculate(system=self)

        if initial_populations is not None:
            for species in initial_populations.keys():
                if species not in self.species:
                    raise KeyError(f'{species} is not a valid species for this system')
        self._initial_populations = initial_populations

        self._empty = empty
        self._Trajectory = None

        with time_this_code() as t_build_state_space:
            self._constitutive_states = self._build_state_space()
        with time_this_code() as t_build_generator_matrix:
            self._generator_matrix = self._build_generator_matrix()
        self.timings: Dict[str, float] = {}
        self.timings['t_build_state_space'] = t_build_state_space.elapsed
        self.timings['t_build_generator_matrix'] = t_build_generator_matrix.elapsed

    def _build_state_space(self):
        subspaces = [generate_subspace(Constraint) for Constraint in self.reaction_network.constraints]
        if len(subspaces) > 1:
            return combine_state_spaces(*subspaces)
        else:
            return np.stack(subspaces[0])

    def _build_generator_matrix(self):
        M, K, N = self.M, self.K, self.N
        reaction_network = self.reaction_network
        base = self._constitutive_states.max() + 1
        states_as_numbers = vector_to_number(self._constitutive_states, base)

        if self._initial_populations is not None:
            self._initial_state = np.array([self._initial_populations[species]
                                            if species in self._initial_populations else 0
                                            for species in self.species])
            self._initial_state_index = binary_search(
                states_as_numbers, vector_to_number(self._initial_state, base))
        else:
            self._initial_state = None
            self._initial_state_index = 0

        G_rows = [i for i in range(M)]
        G_cols = [j for j in range(M)]
        G_values = [0.0 for _ in range(M)]
        for j, state_j in enumerate(self._constitutive_states):
            for k, reaction in enumerate(reaction_network.reaction_matrix):
                # does state_j have the necessary reactants for this reaction?
                if not np.all(state_j[np.where(reaction < 0)] > 0):
                    continue
                state_i = state_j + reaction
                if any([state_i[n] > max_value for n, max_value in reaction_network.max_copy_numbers.items()]):
                    continue
                i = binary_search(states_as_numbers, vector_to_number(state_i, base))
                if i == -1:
                    continue
                #print(f'\n{j} -> {i}\n{list(state_j)}\n{list(reaction)}{self.reaction_network.reactions[k].reaction}\n{list(state_i)}')
                ids = reaction_network.species_in_reaction[k]
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

    def update_rates(self, new_rates):
        """updates `self.rates` and `self.G` with new transition rates"""
        for rate_name, new_rate in new_rates.items():
            for k, reaction in enumerate(self.reaction_network.reactions):
                if reaction.rate_name == rate_name:
                    updated_reaction = reaction._replace(rate=new_rate)
                    self.reaction_network._reactions[k] = updated_reaction
                    self._rates[k] = new_rate
                    break
            else:
                raise KeyError(f'{rate_name} is an unknown rate for this system.')
        # recreate G with new rates
        self._generator_matrix = self._build_generator_matrix()

    def update_constraints(self, *args):
        """"""
        species = args[:-1]
        new_value = args[-1]
        for i, Constraint in enumerate(self.reaction_network._constraints):
            if sorted(species) == sorted(Constraint.species_involved):
                new_constraint = Constraint.constraint[:-1] + str(new_value)
                updated_constraint = Constraint._replace(constraint=new_constraint, value=new_value)
                self.reaction_network._constraints[i] = updated_constraint
                break
        else:
            raise KeyError(f"`{species}` not found in system.reaction_network.constraints")
        self._constitutive_states = self._build_state_space()
        self._generator_matrix = self._build_generator_matrix()

    def _calculate_B_and_OmegaT(self, dt):
        Omega = calculate_Omega(self.G)
        OmegaT = Omega * dt
        B = calculate_B(self.G, self.M, Omega)
        return B, Omega, OmegaT

    def run_simulation(self, N_timesteps=None, dt=1.0):
        with time_this_code() as t_build_B_matrix:
            B, Omega, OmegaT = self._calculate_B_and_OmegaT(dt)
        with time_this_code() as t_run:
            if N_timesteps is None:
                trajectory = self._run_simulation_until_steady_state(OmegaT, B)
            else:
                trajectory = self._run_simulation_N_timesteps(N_timesteps, OmegaT, B)
        self.timings['t_build_B_matrix'] = t_build_B_matrix.elapsed
        self.timings[f't_run_IMU'] = t_run.elapsed

        # use a namedtuple to store trajectory data/metadata cleanly
        Trajectory = namedtuple("Trajectory", ['trajectory', 'method', 'dt', 'N_timesteps', 'rates', 'Omega', 'B', 'Q'])
        self._Trajectory = Trajectory(trajectory=trajectory, method='IMU', dt=dt, N_timesteps=len(trajectory),
                                      rates=self._rates, Omega=Omega, B=B, Q=None)

    def _run_simulation_until_steady_state(self, OmegaT, B):
        trajectory = [np.zeros(self.M)]
        trajectory[0][self._initial_state_index] = 1.0
        ts = 0
        while True:
            p_0 = trajectory[ts]
            trajectory.append(calculate_imu_timestep(p_0, OmegaT, B))
            P_difference = trajectory[ts + 1] - trajectory[ts]
            if P_difference.max() < 1e-12:
                break
            ts += 1
        return np.asarray(trajectory)

    def _run_simulation_N_timesteps(self, N_timesteps, OmegaT, B):
        trajectory = np.empty((N_timesteps, self.M))
        trajectory[0] = np.zeros(self.M)
        trajectory[0, self._initial_state_index] = 1.0
        start, stop = 0, N_timesteps - 1
        for ts in range(start, stop):
            p_0 = trajectory[ts]
            trajectory[ts + 1] = calculate_imu_timestep(p_0, OmegaT, B)
        return trajectory

    def continue_simulation(self, N_timesteps=None, dt=1.0):
        B, Omega, OmegaT = self._calculate_B_and_OmegaT(dt)
        if N_timesteps is None:
            trajectory = self._continue_simulation_until_steady_state(OmegaT, B)
        else:
            trajectory = self._contintue_simulation_N_timesteps(N_timesteps, OmegaT, B)

        Trajectory = namedtuple("Trajectory", ['trajectory', 'method', 'dt', 'N_timesteps', 'rates', 'Omega', 'B', 'Q'])
        self._Trajectory = Trajectory(trajectory=trajectory, method='IMU', dt=dt, N_timesteps=len(trajectory),
                                      rates=self._rates, Omega=Omega, B=B, Q=None)

    def _continue_simulation_until_steady_state(self, OmegaT, B):
        trajectory = [self._Trajectory.trajectory[-1]]
        ts = 0
        while True:
            p_0 = trajectory[ts]
            trajectory.append(calculate_imu_timestep(p_0, OmegaT, B))
            P_difference = trajectory[ts + 1] - trajectory[ts]
            if P_difference.max() < 1e-12:
                break
            ts += 1
        return np.vstack((self._Trajectory.trajectory, trajectory))

    def _contintue_simulation_N_timesteps(self, N_timesteps, OmegaT, B):
        trajectory = np.vstack([self._Trajectory, np.empty((N_timesteps, self.M))])
        start = len(self._Trajectory.trajectory) - 1
        stop = start + N_timesteps
        for ts in range(start, stop):
            p_0 = trajectory[ts]
            trajectory[ts + 1] = calculate_imu_timestep(p_0, OmegaT, B)
        return trajectory


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
            with time_this_code() as t_build_B_matrix:
                Omega = calculate_Omega(self.G)
                B = calculate_B(self.G, self.M, Omega)
            self.timings['t_build_B_matrix'] = t_build_B_matrix.elapsed
            function = calculate_imu_timestep
            args = [Omega*dt, B]
            Q = None
        elif method == 'EXPM':
            with time_this_code() as t_matrix_exponential:
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

        self._Trajectory = self.CMETrajectory(
            trajectory=trajectory, method=method, dt=dt, N_timesteps=len(trajectory),
            rates=self._rates, Omega=Omega, B=B, Q=Q)

    def _run_until_steady_state(self, method, function, args, is_continued_run):
        if is_continued_run:
            trajectory = [self.trajectory[-1]]
        else:
            trajectory = [np.zeros(self.M)]
            trajectory[0][self._initial_state_index] = 1.0

        with time_this_code() as t_run:
            ts = 0
            while True:
                trajectory.append(function(trajectory[ts], *args))
                P_difference = trajectory[ts + 1] - trajectory[ts]
                if P_difference.max() < 1e-12:
                    break
                ts += 1
        self.timings[f't_run_{method}'] = t_run.elapsed
        return np.vstack((self._Trajectory.trajectory, trajectory)) if is_continued_run else np.asarray(trajectory)

    def _run_N_timesteps(self, N_timesteps, method, function, args, is_continued_run):
        if is_continued_run:
            trajectory = np.vstack([self.trajectory, np.empty((N_timesteps, self.M))])
            start = len(self._Trajectory.trajectory) - 1
            stop = start + N_timesteps
        else:
            trajectory = np.empty((N_timesteps, self.M))
            trajectory[0] = np.zeros(self.M)
            trajectory[0, self._initial_state_index] = 1.0
            start, stop = 0, N_timesteps - 1
        with time_this_code() as t_run:
            for ts in range(start, stop):
                trajectory[ts + 1] = function(trajectory[ts], *args)
        self.timings[f't_run_{method}'] = t_run.elapsed
        return trajectory

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
    def Trajectory(self) -> namedtuple:
        """probability trajectory is contained in `Trajectory.trajectory`"""
        return self._Trajectory
    @Trajectory.setter
    def Trajectory(self, value):
        self._Trajectory = value
    @property
    def trajectory(self):
        return self._Trajectory.trajectory
