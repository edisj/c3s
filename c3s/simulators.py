import yaml
import copy
from typing import List, Dict
from collections import namedtuple
from pathlib import Path

import math
import random
import numpy as np
from scipy.sparse.linalg import expm

from .calculations import CalculationsMixin
from .utils import timeit
from .h5io import CMEWriter


class Reactions:
    """meant to be used as a component class via composition for the simulator classes"""

    def __init__(self, config):
        """reads the config file and sets various important attributes"""

        # for reading configs from h5 file
        if isinstance(config, dict):
            self._original_config = config
        else:
            config = Path(config)
            with open(config) as yaml_file:
                self._original_config = yaml.load(yaml_file, Loader=yaml.Loader)
        self._rates, self._reaction_strings, self._reactants, self._products = self._set_rates()
        self._species = self._set_species_vector()
        self._reaction_matrix = self._set_reaction_matrix()
        self._propensity_ids = self._set_reaction_propensities()

    @property
    def rates(self):
        """`self._rates` is len(K) List[List[str, float]] where k'th element gives
        the name and value of the rate constant for the k'th reaction
        """
        return dict(self._rates)
    @rates.setter
    def rates(self, value):
        self._rates = value

    @property
    def species(self) -> List[str]:
        """`self.species` is len(N) List[str] where n'th element is the name of the n'th species"""
        return self._species
    @species.setter
    def species(self, value):
        self._species = value

    @property
    def reaction_matrix(self) -> np.ndarray:
        """`self.reaction_matrix` is shape(K,N) array where the [k,n] element
        gives the change in the n'th species for the k'th reaction
        """
        return self._reaction_matrix
    @reaction_matrix.setter
    def reaction_matrix(self, value):
        self._reaction_matrix = value

    @property
    def propensity_ids(self):
        """`self._propensenity_ids` is len(K) List[List[int]] whose k'th element
         gives the indices of `self.species` that are involved in the k'th reaction
         """
        return self._propensity_ids
    @propensity_ids.setter
    def propensity_ids(self, value):
        self._propensity_ids = value

    def _set_rates(self):
        rates, reaction_strings, reactants, products = [], [], [], []
        # deepcopy because update_rates() would otherwise change rates in self._original_config
        config_data = copy.deepcopy(self._original_config)
        for reaction, rate_list in config_data['reactions'].items():
            rates.append(rate_list)
            reaction_strings.append(reaction)
            reactants.append(reaction.replace(' ', '').split('->')[0].split('+'))
            products.append(reaction.replace(' ', '').split('->')[1].split('+'))
        return rates, reaction_strings, reactants, products

    def _set_species_vector(self):
        species: List[str] = []
        for k, (reactants, products) in enumerate(zip(self._reactants, self._products)):
            # len(reactants) is not necessarily = len(products) so we have to loop over each
            # TODO: need to add check and warning for birth process without max_popoulations
            for molecule in reactants:
                species.append(molecule)
            for molecule in products:
                species.append(molecule)
        while '0' in species:
            species.remove('0')
        # remove duplicates and sort
        species = sorted(list(set(species)))
        return species

    def _set_reaction_matrix(self):
        N_reactions = len(self._reactants)
        N_species = len(self._species)
        reaction_matrix = np.zeros(shape=(N_reactions, N_species), dtype=np.int32)
        for reaction, reactants, products in zip(reaction_matrix, self._reactants, self._products):
            for n, species in enumerate(self._species):
                # if this species in both a product and reactant, net effect is 0
                if species in reactants:
                    reaction[n] += -1
                if species in products:
                    reaction[n] += 1
        return reaction_matrix

    def _set_reaction_propensities(self):
        reaction_matrix = self._reaction_matrix
        N, K = len(self._species), len(reaction_matrix)
        propensity_ids = [[n for n in range(N) if reaction_matrix[k, n] < 0] for k in range(K)]
        return propensity_ids

    def print_propensities(self):
        """generates a readable list of the propensity of each reaction"""

        propensity_strings: List[str] = []
        for propensity_ids, rate in zip(self._propensity_ids, self._rates):
            transition_rate = rate[0]
            for n in propensity_ids:
                transition_rate += f'c^{self._species[n]}'
            propensity_strings.append(transition_rate)

        return propensity_strings


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
                 config=None,
                 initial_state=None,
                 initial_populations=None,
                 max_populations=None,
                 empty=False,
                 low_memory=False):
        """
        Parameters
        ----------
        config :
            path to yaml config file that specifies chemical reactions and elementary rates
        initial_state : List[int] or array
            initial collapsed state vector, if initial populations is not given
        initial_populations : dict, default=None
            dictionary of initial species populations
            if a species population is not specified, it's initial population is taken to be 0
        max_populations : dict, default=None
            maximum allowable populaation for some species
        empty : bool, default=False
            flag to leave attributes empty in the case of reading from file
        low_memory : bool, default=False
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
    def states(self):
        return self._constitutive_states
    @states.setter
    def states(self, value):
        self._constitutive_states = value

    @property
    def G(self):
        """the generator matrix is an MxM matrix where M is the
        total number of states given by `len(self.constitutive_states)`"""
        return self._generator_matrix

    @property
    def trajectory(self):
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
        """constructs the generator matrix"""

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

    def _set_propagator_matrix(self, dt=1):
        # dont really use this
        with timeit() as matrix_exponential:
            Q = expm(self._generator_matrix * dt)
        self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
        self.Q = Q
        self._dt = dt

    def run(self, start, stop, dt=1, overwrite=False, continued=False, run_name=None):
        """runs the chemical master equation simulation

        Parameters
        ----------
        start : int or float
            initial time value
        stop : int or float
            final time value
        dt : int or float, default=1
            size of timestep that multiplies into generator matrix
        overwrite : bool, default=False
            set to `True` to rerun a simulation from scratch
        continued : bool, default=False
            set to `True` to concatenate separate trajectory segemnts
        run_name : str, default=None

        """

        if self._trajectory is not None and not overwrite:
            if not continued:
                raise ValueError("Data from previous run found in `self.trajectory`. "
                                 "To write over this data, set the `overwrite=True`")

        self._run(start, stop, dt, overwrite, continued, run_name)

    def _run(self, start, stop, dt, overwrite, continued, run_name):

        # using np.round to avoid floating point precision errors
        n_timesteps = int(np.round((stop - start) / dt))
        M = self.M

        with timeit() as matrix_exponential:
            if self._low_memory:
                Q = expm(self._build_generator_matrix() * dt)
            else:
                Q = expm(self._generator_matrix * dt)
        self.timings['t_matrix_exponential'] = matrix_exponential.elapsed

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

    def write(self, filename, mode='r+', trajectory_name=None):
        """writes simulation data to an hdf5 file"""

        if self.trajectory is None:
            raise ValueError("no data in `self.trajectory`")
        if not Path(filename).exists():
            # if this is a fresh file
            self._write_system_info(filename, mode='x')

        self._write_trajectory(filename, mode, trajectory_name)


class SimulatorBase:
    """This is the parent class of all the simulators."""

    np.seterr(under='raise', over='raise')

    def __init__(self, cfg,
                 initial_state=None, initial_populations=None, max_populations=None, empty=False):
        """Reads a yaml config file that specifies the chemical reactions and kinetic rates for the chemical system
        and sets important attributes.

        Parameters
        ----------
        cfg : str
            Path to yaml config file that specifies chemical reactions and kinetic rates
        initial_state : list of int or array_like
        initial_populations : dict, default=None
            The initial population a particular species. If a species population is
            not specified, it's initial population is taken to be 0.
        max_populations : dict

        """

        self._config_path = cfg
        self._config_dictionary = None
        self._initial_populations = initial_populations
        self._initial_state = initial_state
        if self._initial_state and self._initial_populations:
            raise ValueError("Do not specify both the `initial_state` and `initial_populations` parameters. "
                             "Use one or the other.")
        self._max_populations = max_populations
        self._reactants = None
        self._products = None
        self._rates = None
        self._species = None
        self._propensity_indices = None
        self._reaction_matrix = None
        self._results = None
        self._dt = None

        # now that the setup is taken care of, begin filling the data structures
        with open(self._config_path) as file:
            self._config_dictionary = yaml.load(file, Loader=yaml.Loader)

        self._set_rates()
        self._set_species_vector()
        self._set_initial_state()
        self._set_reaction_matrix_and_propensity_indices()

        # dictionary to hold timings of various codeblocks for benchmarking
        self.timings: Dict[str, float] = {}

    def _set_rates(self):
        """create `self.rates` which is len(K) List[List[str, int]] where k'th element gives
        the name and value of the rate constant for the k'th reaction"""

        reactants = []
        products = []
        rates = []
        # need to use deepcopy because `self.update_rates()` will change rates in self._config_dictionary
        config_data = copy.deepcopy(self._config_dictionary)
        for reaction, rate_list in config_data['reactions'].items():
            reactants.append(reaction.replace(' ', '').split('->')[0].split('+'))
            products.append(reaction.replace(' ', '').split('->')[1].split('+'))
            rates.append(rate_list)

        self._reactants = reactants
        self._products = products
        self._rates = rates

    def _set_species_vector(self):
        """creates `self.species` which is len(N) List[str] where n'th element is the name of the n'th species"""

        species: List[str] = []
        for k, (reactants, products) in enumerate(zip(self._reactants, self._products)):
            # len(reactants) is not necessarily = len(products) so we have to loop over each
            # TODO: need to add check and warning for birth process without max_popoulations
            for molecule in reactants:
                species.append(molecule)
            for molecule in products:
                species.append(molecule)

        while '0' in species:
            species.remove('0')
        # remove duplicates and sort
        species = sorted(list(set(species)))

        self._species = species

    def _set_initial_state(self):
        """sets the `self._initial_state` attribute that specifies the vector of species counts at t=0"""

        if self._initial_state:
            assert len(self._initial_state) == len(self._species)
            # initial state was specified by the user
            return

        for species in self._initial_populations.keys():
            if species not in self._species:
                raise KeyError(f'{species} is not a valid species. It must be in one of the '
                               f'chemical reactions specified in the config file.')
        if self._max_populations:
            for species in self._max_populations.keys():
                if species not in self._species:
                    raise KeyError(f'{species} is not a valid species. It must be in one of the '
                                   f'chemical reactions specified in the config file.')

        initial_state = [self._initial_populations[species]
                         if species in self._initial_populations else 0
                         for species in self._species]

        self._initial_state = initial_state

    def _set_reaction_matrix_and_propensity_indices(self):
        """has 2 jobs:
            - creates `self.reaction_matrix` which is shape(K,N) array where the [k,n] element
              gives the change in the n'th species for the k'th reaction
            - creates `self._propensenity_species_ids` which is len(K) List[List[int]] which gives the indices
              of `self.species` that are involved in the k'th reaction
        """

        N_reactions = len(self._reactants)
        N_species = len(self._species)
        reaction_matrix = np.zeros(shape=(N_reactions, N_species), dtype=np.int32)
        for reaction, reactants, products in zip(reaction_matrix, self._reactants, self._products):
            for n, species in enumerate(self._species):
                # if this species in both a product and reactant, net effect is 0
                if species in reactants:
                    reaction[n] += -1
                if species in products:
                    reaction[n] += 1

        self._reaction_matrix = reaction_matrix
        self._propensity_indices = [[n for n in range(N_species) if reaction_matrix[k,n] < 0]
                                    for k in range(len(reaction_matrix))]

    def get_propensity_strings(self):
        """creates a readable list of the propensity of each reaction"""

        propensity_strings: List[str] = []
        for propensity_ids, rate in zip(self._propensity_indices, self._rates):
            transition_rate = rate[0]
            for n in propensity_ids:
                transition_rate += f'c^{self._species[n]}'
            propensity_strings.append(transition_rate)

        return propensity_strings

    def reset_rates(self):
        """resets `self.rates` to the original values from the config file"""
        rates_from_config = [rate for rate in self._config_dictionary['reactions'].values()]
        original_rates = {rate[0]: rate[1] for rate in rates_from_config}
        self.update_rates(original_rates)

    def update_rates(self, new_rates):
        """the child class decides what to do with this"""
        pass

    def run(self, *args, **kwargs):
        """runs the simulator of the child class"""
        pass

    @property
    def species(self) -> List[str]:
        return self._species
    @species.setter
    def species(self, value):
        self._species = value
    @property
    def reaction_matrix(self) -> np.ndarray:
        return self._reaction_matrix
    @reaction_matrix.setter
    def reaction_matrix(self, value):
        self._reaction_matrix = value
    @property
    def rates(self):
        return dict(self._rates)
    @rates.setter
    def rates(self, value):
        self._rates = value


class Gillespie(SimulatorBase):
    """The Gillespie stochastic simulation algorithm (SSA)."""

    def __init__(self, cfg, initial_state=None, initial_populations=None, max_populations=None, empty=False):
        """Uses Gillespie's stochastic simulation algorithm to generate a trajectory of a random walker that is
        defined by a molecular population vector.

        Parameters
        ----------
        cfg : str
            Path to yaml config file that specifies chemical reactions and kinetic rates
        initial_state : list of int or array_like
        initial_populations : dict, default=None
            The initial population a particular species. If a species population is
            not specified, it's initial population is taken to be 0.
        max_populations : dict
        empty : bool

        """

        super(Gillespie, self).__init__(cfg=cfg,
                                        initial_state=initial_state,
                                        initial_populations=initial_populations,
                                        max_populations=max_populations,
                                        empty=empty)

    def _get_propensity_vector(self, currState):
        """"""
        reaction_propensities = [np.prod(currState[indices])*rate[1]
                                 for indices, rate in zip(self._propensity_indices, self._rates)]

        return np.array(reaction_propensities, dtype=np.int32)

    def _get_next_state(self, currState, propensity_vector):
        """"""

        # k selects the index of which reaction was sampled to fire
        k = self._sample_categorical(propensity_vector)
        reaction = self._reaction_matrix[k]
        nextState = currState + reaction
        return nextState

    def _sample_categorical(self, propensity_vector):
        """Samples a categorical distribution.

        The idea for sampling from a discrete categorical distribution goes as follows:
            - imagine a number line from 0-1, call it L
            - take the probability of each possible outcome and fill the corresponding section on the line.
              for example, if event p(A) = 25%, then the numbers 0-0.25 correspond to event A.
              call the section that the k'th event takes up on the line L_k
            - sample a uniform random number, u,  between 0 and 1
            - find the smallest index k such that u < L_1 + ... + L_k + ... + L_K
            - k gives the index of which event was sampled

        """
        u = random.uniform(0, 1)
        # normalize for probabilities
        event_probabilities = propensity_vector / np.sum(propensity_vector)
        # np.cumsum() does the job nicely
        k = np.argmax(np.cumsum(event_probabilities) > u)
        return k

    def _sample_holding_time(self, propensity_vector):
        """"""
        lambda_ = np.sum(propensity_vector)
        u = random.uniform(0, 1)
        h = -(1 / lambda_) * math.log(1 - u)
        return h

    def update_rates(self, new_rates):
        """updates `self.rates` and with new elementary transition rates"""

        for new_rate_string, new_rate_value in new_rates.items():
            for k, old_rate in enumerate(self._rates):
                if old_rate[0] == new_rate_string:
                    self._rates[k][1] = new_rate_value
                    break
            else:
                raise KeyError(f'{new_rate_string} is not a valid rate for this system. The new rate'
                               f'must be one of the rates specified in the original config file.')

    def run(self, N_timesteps, run_name=None, overwrite=False):
        """Runs the stochastic simulation algorithm.

        Parameters
        ----------
        N_timesteps : int
        overwrite   : bool, default=False

        """

        if self._results is not None and not overwrite:
            raise ValueError("Data from previous run found in `self.trajectory`. "
                             "To write over this data, set the `overwrite=True`")

        N_species = len(self._species)
        # let's use a named tuple because it's very Pythonic...
        Trajectory = namedtuple("Trajectory", ["states", "times"])
        sequence_of_states = np.empty(shape=(N_timesteps, N_species), dtype=np.int32)
        times = np.empty(shape=N_timesteps, dtype=np.float64)
        trajectory = Trajectory(sequence_of_states, times)

        currTime = 0
        currState = np.array(self._initial_state, dtype=np.int32)
        for ts in range(N_timesteps):
            trajectory.states[ts] = currState
            propensity_vector = self._get_propensity_vector(currState)
            holding_time = self._sample_holding_time(propensity_vector)
            nextState = self._get_next_state(currState, propensity_vector)
            currTime += holding_time
            trajectory.times[ts] = currTime
            # set current state to the new state and proceed along the journey
            currState = nextState

        self._results = trajectory

    def run_many_iterations(self, N_iterations, N_timesteps, run_name=None):
        """

        Parameters
        ----------
        N_iterations : int
        N_timesteps  : int

        """

        N_species = len(self._species)
        trajectories = np.empty(shape=(N_iterations, N_timesteps, N_species), dtype=np.int32)
        times = np.empty(shape=(N_iterations, N_timesteps), dtype=np.float64)

        for i in range(N_iterations):
            self.run(N_timesteps, run_name=run_name, overwrite=True)
            trajectories[i] = self.trajectory.states
            times[i] = self.trajectory.times

        self._results = (trajectories, times)

    @property
    def trajectory(self):
        return self._results
    @trajectory.setter
    def trajectory(self, value):
        self._results = value
    @property
    def Trajectories(self):
        return self._results
    @Trajectories.setter
    def Trajectories(self, value):
        self._results = value
