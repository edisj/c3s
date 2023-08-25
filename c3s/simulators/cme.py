from typing import List, Dict

import numpy as np
from scipy.sparse.linalg import expm

from .reactions import ReactionNetwork
from ..calculations import CalculationsMixin
from ..math_utils import combine_state_spaces, vector_to_number, binary_search
from ..utils import timeit
from ..sparse_matrix import SparseMatrix


class ChemicalMasterEquation(CalculationsMixin):
    """
    Simulator class of the Chemical Master Equation (CME).

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
                 config=None,
                 initial_state=None,
                 initial_populations=None,
                 max_populations=None,
                 empty=False):
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
                maximum allowable population for some species
            empty:
                flag to leave attributes empty in the case of reading from file
        """

        self.reaction_network = ReactionNetwork(config)
        self.species = self.reaction_network.species
        self._rates = self.reaction_network.rates
        if initial_state and initial_populations:
            raise ValueError("Do not specify both the `initial_state` and `initial_populations` parameters. "
                             "Use one or the other.")
        self._initial_state = initial_state
        self._initial_populations = initial_populations
        self._max_populations = max_populations
        self._set_initial_state()
        self._constitutive_states = None
        self._generator_matrix = None
        self._nonzero_G_elements = None
        self._trajectory = None

        # dictionary to hold timings of various codeblocks for benchmarking
        self.timings: Dict[str, float] = {}
        if not empty:
            self._set_constitutive_states()
            self._set_generator_matrix()

    @property
    def states(self) -> np.ndarray:
        return self._constitutive_states
    @states.setter
    def states(self, value):
        self._constitutive_states = value
    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory
    @trajectory.setter
    def trajectory(self, value):
        self._trajectory = value

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
    @property
    def G(self) -> np.ndarray:
        """G is the MxM transition rate matrix in column format"""
        return self._generator_matrix

    def _set_initial_state(self):
        """sets the `self.initial_state` attribute that specifies the vector of species counts at t=0"""

        if self._initial_state:
            assert len(self._initial_state) == self.N
            # initial state was specified by the user
            return
        if self._max_populations:
            for species in self._max_populations.keys():
                if species not in self.species:
                    raise KeyError(f'{species} is not a valid species. It must be in one of the '
                                   f'chemical reactions specified in the config file.')
        if self._initial_populations:
            for species in self._initial_populations.keys():
                if species not in self.species:
                    raise KeyError(f'{species} is not a valid species. It must be in one of the '
                                   f'chemical reactions specified in the config file.')
            initial_state = [self._initial_populations[species]
                             if species in self._initial_populations else 0
                             for species in self.species]
            self._initial_state = initial_state

    def _set_constitutive_states(self):
        with timeit() as set_constitutive_states:
            self._constitutive_states = self._build_state_space()
        self.timings['t_build_states'] = set_constitutive_states.elapsed

    def _set_generator_matrix(self):
        with timeit() as set_G_matrix:
            self._rows, self._cols, self.values = self._build_generator_matrix()
        self.timings['t_build_G'] = set_G_matrix.elapsed

    def _build_state_space(self):
        subspaces = [self._generate_subspace(constraint) for constraint in self.reaction_network.constraints]
        if len(subspaces) > 1:
            return combine_state_spaces(*subspaces)
        else:
            return np.stack(subspaces[0])

    def _generate_subspace(self, constraint):
        lim = constraint.value
        N = len(constraint.species_involved)
        if constraint.separator == '<=':
            subspace = np.arange(lim+1)
            return subspace.reshape(subspace.size, -1)
        elif constraint.separator == '=':
            subspace = [[n // (lim+1)**i % (lim+1) for i in range(N)] for n in range((lim+1)**N)]
            subspace = np.stack([np.flip(state) for state in subspace if sum(state) == lim])
            return subspace

    def _build_generator_matrix(self):

        M, K, N = self.M, self.K, self.N

        G_rows = [i for i in range(M)]
        G_cols = [j for j in range(M)]
        G_values = [0 for _ in range(M)]
        # gives which elements of G the k'th reaction is responsible for
        k_to_G_map = {k: [] for k in range(K)}

        base = self._constitutive_states.max() + 1
        states = vector_to_number(self._constitutive_states, N, base)
        reactions = vector_to_number(self.reaction_network.reaction_matrix, N, base)
        for j, state_j in enumerate(states):
            for k, reaction in enumerate(reactions):
                state_i = state_j + reaction
                # find index of state_i in state space
                i = binary_search(states, state_i)
                if not i:
                    # state_j + reaction_k was not in state space
                    continue
                # indices of which species are involved in k'th reaction
                ids = self.reaction_network.species_in_reaction[k]
                # the elementary transition rate of the k'th reaction
                rate = self._rates[k]
                # the combinatorial factor associated for the k'th reaction for the j'th state
                h = np.prod([self.states[j,n] for n in ids])
                # overall reaction propensity from the j'th state
                propensity = h * rate
                G_rows.append(i)
                G_cols.append(j)
                G_values.append(propensity)
                G_values[j] -= propensity
                k_to_G_map[k].append((i,j))

        self._k_to_G_map = k_to_G_map
        G = SparseMatrix(np.array(G_rows), np.array(G_cols), np.array(G_values))
        return G

    def run(self, N_timesteps, dt=1, overwrite=False, continued=False):
        """runs the chemical master equation simulation

        Args:
            N_timesteps:
                number of timesteps
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

        self._run(N_timesteps, dt, overwrite, continued)

    def _run(self, N_timesteps, dt, overwrite, continued):

        M = self.M
        if self.Q is None:
            Q = expm(self._generator_matrix*dt)
            self.Q = Q
        else:
            Q = self.Q

        trajectory = np.empty(shape=(N_timesteps, M), dtype=np.float64)
        if continued:
            trajectory[0] = Q.dot(self._trajectory[-1])
        else:
            # fixing initial probability to be 1 in the initial state
            trajectory[0] = np.zeros(shape=M, dtype=np.float64)
            trajectory[0, 0] = 1.0

        with timeit() as run_time:
            for ts in range(N_timesteps - 1):
                trajectory[ts + 1] = Q.dot(trajectory[ts])
        self.timings['t_run'] = run_time.elapsed

        self._trajectory = np.vstack([self._trajectory, trajectory]) if continued else trajectory
        self._dt = dt

    def update_rates(self, new_rates):
        """updates `self.rates` and `self.G` with new transition rates"""
        for rate_name, new_rate in new_rates.items():
            for reaction in self.reaction_network.reactions:
                if reaction.rate_name == rate_name:
                    reaction.rate = new_rate
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

    '''def print_constitutive_states(self):
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

        M, K = self.M, self.K
        readable_G = [['0' for _ in range(M)] for _ in range(M)]
        propensity_strings = self.reaction_network.print_propensities()
        for k in range(K):
            for idx in self._nonzero_G_elements[k]:
                i,j = idx
                readable_G[i][j] = propensity_strings[k]
        for j in range(M):
            diagonal = '-('
            for i in range(M):
                if readable_G[i][j] != '0':
                    diagonal += f'{readable_G[i][j]} + '
            readable_G[j][j] = diagonal[:-3] + ')'

        return readable_G'''

    def __build_constitutive_states_OLD(self):
        """iteratively generates all possible constitutive states from the intial state"""

        constitutive_states = [self._initial_state]
        population_limits = [
            (self.species.index(species), max_count)
            for species, max_count in self._max_populations.items()
        ] if self._max_populations else False

        self._nonzero_G_elements = {k: [] for k in range(len(self.reaction_network.reaction_matrix))}

        # newly_added keeps track of the most recently accepted states
        newly_added_states = [np.array(self._initial_state)]
        while True:
            accepted_candidate_states = []
            for state in newly_added_states:
                i = int(np.argwhere(np.all(constitutive_states == state, axis=1)))
                # the idea here is that for each of the recently added states,
                # we iterate through each reaction to see if a transition is possible
                for k, reaction in enumerate(self.reaction_network.reaction_matrix):
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

        self._constitutive_states = np.array(constitutive_states, dtype=np.int32)

    def __build_generator_matrix_OLD(self):
        M = self.M
        G = np.zeros(shape=(M,M), dtype=float)
        for k, value in self._nonzero_G_elements.items():
            for idx in value:
                i,j = idx
                # the indices of the species involved in the reaction
                n_ids = self.reaction_network.species_in_reaction[k]
                # h is the combinatorial factor for number of reactions attempting to fire
                # At the moment this assumes maximum stoichiometric coefficient of 1
                # TODO: generalize h for any coefficient
                state_j = self._constitutive_states[j]
                h = np.prod([state_j[n] for n in n_ids])
                rate = self.reaction_network.reactions[k].rate
                reaction_propensity = h * rate
                G[i,j] = reaction_propensity
        for i in range(M):
            # fix the diagonal to be the negative sum of the column
            G[i,i] = -np.sum(G[:,i])
        return G
