from typing import List, Dict
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm

from .reactions import Reactions
from ..calculations import CalculationsMixin
from ..utils import timeit
from ..h5io import CMEWriter


class ChemicalMasterEquation(CalculationsMixin):
    """
    Simulator class of the Chemical Master Equationn (CME).

    Uses the Chemical Master Equation to numerically integrate the time
    evolution of the probability trajectory of a chemical system.

    The only input required by the user is the config file that specifies the elementary chemical reactions
    and kinetic rates, and the initial nonzero population numbers of each species. The constructor will
    build the full constitutive state space in `self.constitutive_states` and the generator matrix in
    `self.G`. To run the simulator, call the `self.run()` method with start, stop, and dt
    arguments. The full P(t) trajectory will be stored in `self.trajectory`.

    """

    np.seterr(under='raise', over='raise')

    def __init__(self,
                 config: Path = None,
                 initial_state: np.ndarray = None,
                 initial_populations = None,
                 max_populations = None,
                 empty: bool = False,
                 low_memory: bool = False):
        """
        Args:
            config:
                path to yaml config file that specifies chemical reactions and elementary rates
            initial_state:
                initial collapsed state vector, if initial populations is not given
            initial_populations:
                dictionary of initial species populations
                if a species population is not specified, it's initial population is taken to be 0
            max_populations:
                maximum allowable populaation for some species
            empty:
                flag to leave attributes empty in the case of reading from file
            low_memory:
                flag to use 32 bit precision and to not save constitutive states in memeory
        """

        self.reactions = Reactions(config)
        self.species = self.reactions.species
        self.rates = self.reactions.rates
        self._rates = self.reactions._rates
        if initial_state and initial_populations:
            raise ValueError("Do not specify both the `initial_state` and `initial_populations` parameters. "
                             "Use one or the other.")
        self._initial_state = initial_state
        self._initial_populations = initial_populations
        self._max_populations = max_populations
        self._set_initial_state()
        self._low_memory = low_memory
        self._array_dtype = np.float32 if self._low_memory else np.float64

        self._constitutive_states = None
        self._generator_matrix = None
        self._nonzero_G_elements = None
        self._trajectory = None
        self.Q = None

        # dictionary to hold timings of various codeblocks for benchmarking
        self.timings: Dict[str, float] = {}

        if not empty:
            with timeit() as set_constitutive_states:
                self._build_constitutive_states()
            if not self._low_memory:
                with timeit() as set_generator_matrix:
                    self._set_generator_matrix()
            self.timings['t_set_constitutive_states'] = set_constitutive_states.elapsed

    @property
    def states(self) -> np.ndarray:
        return self._constitutive_states
    @states.setter
    def states(self, value):
        self._constitutive_states = value

    @property
    def G(self) -> np.ndarray:
        """the generator matrix is an MxM matrix where M is the
        total number of states given by `len(self.constitutive_states)`"""
        return self._generator_matrix

    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory
    @trajectory.setter
    def trajectory(self, value):
        self._trajectory = value

    def _set_initial_state(self):
        """sets the `self.initial_state` attribute that specifies the vector of species counts at t=0"""

        if self._initial_state:
            assert len(self._initial_state) == len(self.species)
            # initial state was specified by the user
            return

        for species in self._initial_populations.keys():
            if species not in self.species:
                raise KeyError(f'{species} is not a valid species. It must be in one of the '
                               f'chemical reactions specified in the config file.')
        # dont remember why I put this in this method
        if self._max_populations:
            for species in self._max_populations.keys():
                if species not in self.species:
                    raise KeyError(f'{species} is not a valid species. It must be in one of the '
                                   f'chemical reactions specified in the config file.')

        initial_state = [self._initial_populations[species]
                         if species in self._initial_populations else 0
                         for species in self.species]

        self._initial_state = initial_state

    def _build_constitutive_states(self):
        """iteratively generates all possible constitutive states from the intial state"""

        constitutive_states = [self._initial_state]
        population_limits = [
            (self.species.index(species), max_count)
            for species, max_count in self._max_populations.items()
        ] if self._max_populations else False

        self._nonzero_G_elements = {k: [] for k in range(len(self.reactions.reaction_matrix))}

        # newly_added keeps track of the most recently accepted states
        newly_added_states = [np.array(self._initial_state)]
        while True:
            accepted_candidate_states = []
            for state in newly_added_states:
                i = int(np.argwhere(np.all(constitutive_states == state, axis=1)))
                # the idea here is that for each of the recently added states,
                # we iterate through each reaction to see if a transition is possible
                for k, reaction in enumerate(self.reactions.reaction_matrix):
                    # gives a boolean array for which reactants are required
                    reactants_required = np.argwhere(reaction < 0).T
                    reactants_available = state > 0
                    # true if this candidate state has all of the reactants available for the reaction
                    if np.all(reactants_available[reactants_required]):
                        # apply the reaction and add the new state into our list of constitutive
                        # states only if it is a new state that has not been previously visited
                        new_candidate_state = state + reaction
                        is_actually_new_state = list(new_candidate_state) not in constitutive_states
                        does_not_exceed_max_population = all([new_candidate_state[i] <= max_count for i, max_count
                                                              in population_limits]) if population_limits else True
                        if does_not_exceed_max_population:
                            if is_actually_new_state:
                                j = len(constitutive_states)
                                accepted_candidate_states.append(new_candidate_state)
                                constitutive_states.append(list(new_candidate_state))
                            else:
                                j = int(np.argwhere(np.all(constitutive_states == new_candidate_state, axis=1)))
                            self._nonzero_G_elements[k].append((j,i))

            # replace the old set of new states with new batch
            newly_added_states = accepted_candidate_states
            # once we reach the point where no new states are accessible we terminate
            if not newly_added_states:
                break

        self.M = len(constitutive_states)
        self._constitutive_states = np.array(constitutive_states, dtype=np.int32)

    def _set_generator_matrix(self):
        self._generator_matrix = self._build_generator_matrix()

    def _build_generator_matrix(self):

        M = self.M
        K = len(self.reactions.reaction_matrix)
        G = np.zeros(shape=(M, M), dtype=self._array_dtype)
        for k, value in self._nonzero_G_elements.items():
            for idx in value:
                i,j = idx
                # the indices of the species involved in the reaction
                n_ids = self.reactions.propensity_ids[k]
                # h is the combinatorial factor for number of reactions attempting to fire
                # At the moment this assumes maximum stoichiometric coefficient of 1
                # TODO: generalize h for any coefficient
                state_j = self._constitutive_states[j]
                h = np.prod([state_j[n] for n in n_ids])
                # lambda_ is the elementary reaction rate for the k'th reaction
                lambda_ = self.reactions._rates[k][1]
                reaction_propensity = h * lambda_
                G[i,j] = reaction_propensity
        for i in range(M):
            # fix the diagonal to be the negative sum of the column
            G[i,i] = -np.sum(G[:,i])

        return G

    def _set_propagator_matrix(self, dt):
        # dont really use this
        with timeit() as matrix_exponential:
            Q = expm(self._generator_matrix * dt)
        self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
        self.Q = Q
        self._dt = dt

    def run(self, start: float, stop:float , dt:float = 1,
            overwrite: bool = False, continued: bool = False):
        """runs the chemical master equation simulation

        Args:
            start:
                initial time value
            stop:
                final time value
            dt (1):
                size of timestep that multiplies into generator matrix
            overwrite (False):
                set to `True` to rerun a simulation from scratch
            continued (False):
                set to `True` to concatenate separate trajectory segments
        """

        if self._trajectory is not None and not overwrite:
            if not continued:
                raise ValueError("Data from previous run found in `self.trajectory`. "
                                 "To write over this data, set the `overwrite=True`")

        self._run(start, stop, dt, overwrite, continued)

    def _run(self, start, stop, dt, overwrite, continued):

        # using np.round to avoid floating point precision errors
        n_timesteps = int(np.round((stop - start) / dt))
        M = self.M

        #if self._low_memory:
            #Q = expm(self._build_generator_matrix() * dt)
        #else:
        if self.Q is None:
            Q = expm(self._generator_matrix * dt)
            self.Q = Q
        else:
            Q = self.Q

        trajectory = np.empty(shape=(n_timesteps, M), dtype=self._array_dtype)
        if continued:
            trajectory[0] = Q.dot(self._trajectory[-1])
        else:
            # fixing initial probability to be 1 in the intitial state
            trajectory[0] = np.zeros(shape=M, dtype=self._array_dtype)
            trajectory[0, 0] = 1.0

        with timeit() as run_time:
            for ts in range(n_timesteps - 1):
                trajectory[ts + 1] = Q.dot(trajectory[ts])
        self.timings['t_run'] = run_time.elapsed

        self._trajectory = np.vstack([self._trajectory, trajectory]) if continued else trajectory
        self._dt = dt

    def reset_rates(self):
        """resets `self.rates` to the values of the original config file"""

        rates_from_config = [rate for rate in self.reactions._original_config['reactions'].values()]
        original_rates = {rate[0]: rate[1] for rate in rates_from_config}
        self.update_rates(original_rates)

    def update_rates(self, new_rates):
        """updates `self.rates` and `self.G` with new transition rates"""

        for new_rate_string, new_rate_value in new_rates.items():
            for k, old_rate in enumerate(self._rates):
                if old_rate[0] == new_rate_string:
                    if old_rate[1] == new_rate_value:
                        propensity_adjustment_factor = 1
                    else:
                        #TODO: handle the old_rate=0 case
                        propensity_adjustment_factor = new_rate_value / old_rate[1]
                    # make sure to do this after saving the propensity factor
                    self._rates[k][1] = new_rate_value
                    self.rates[new_rate_string] = new_rate_value
                    # the generator matrix also changes when the rates change
                    G_elements_affected = self._nonzero_G_elements[k]
                    for idx in G_elements_affected:
                        i = tuple(idx)
                        self._generator_matrix[i] = self._generator_matrix[i] * propensity_adjustment_factor
                    break
            else:
                raise KeyError(f'{new_rate_string} is not a valid rate for this system. Valid rates'
                               f'are listed in `self.rates`.')
        # need to redo the diagonal elements as well
        for m in range(len(self._generator_matrix)):
            self._generator_matrix[m,m] = 0
            self._generator_matrix[m,m] = -np.sum(self._generator_matrix[:, m])

    def write(self, filename, mode='r+', trajectory_name=None):
        """writes simulation data to an hdf5 file"""

        if not Path(filename).exists():
            # if this is a fresh file
            self._write_system_info(filename, mode='x')

        if trajectory_name:
            if self.trajectory is None:
                raise ValueError("no data in `self.trajectory`")
            self._write_trajectory(filename, mode, trajectory_name)

    def _write_system_info(self, filename, mode):
        with CMEWriter(filename, system=self, mode=mode) as W:
            # basic reaction info
            W._dump_config()
            if not self._low_memory:
                W._create_dataset('constitutive_states', data=self.states)
            for k, indices in self._nonzero_G_elements.items():
                W._create_dataset(f'nonzero_G_elements/{k}', data=np.array(indices))
            for species, count in self._initial_populations.items():
                W._create_dataset(f'initial_populations/{species}', data=np.array(count))
            if self._max_populations:
                for species, count in self._max_populations.items():
                    W._create_dataset(f'max_populations/{species}', data=np.array(count))

    def _write_trajectory(self, filename, mode, trajectory_name):
        with CMEWriter(filename, system=self, mode=mode) as W:
            traj_group = W._require_group('trajectories')
            if trajectory_name is None:
                trajectory_name = f'trajectory00{len(traj_group) + 1}'
            W._create_dataset(f'trajectories/{trajectory_name}/trajectory', data=self.trajectory)
            for rate, value in self.rates.items():
                rate_group = W._create_group(f'trajectories/{trajectory_name}/rates/{rate}')
                W._set_attr(rate_group, name='value', value=value)

    def print_constitutive_states(self):
        """generates a convenient readable list of the constitutive states"""

        constitutive_states_strings: List[List[str]] = []
        for state in self._constitutive_states:
            word = []
            for population_number, species in zip(state, self.species):
                word.append(f'{population_number}{species}')
            constitutive_states_strings.append(word)

        return constitutive_states_strings

    def print_generator_matrix(self):
        """generates a convenient readable generator matrix with string names"""

        M = len(self.G)
        readable_G = [['0' for _ in range(M)] for _ in range(M)]
        propensity_strings = self.reactions.print_propensities()
        for k in range(len(self.reactions.reaction_matrix)):
            for idx in self._nonzero_G_elements[k]:
                i,j = idx
                readable_G[i][j] = propensity_strings[k]
        for j in range(M):
            diagonal = '-('
            for i in range(M):
                if readable_G[i][j] != '0':
                    diagonal += f'{readable_G[i][j]} + '
            readable_G[j][j] = diagonal[:-3] + ')'

        return readable_G
