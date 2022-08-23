import numpy as np
import yaml
import copy
from scipy.sparse import linalg
from pathlib import Path
from .utils import timeit, ProgressBar, slice_tasks_for_parallel_workers
from .calculations import CalculationsMixin
from .plotting import PlottingMixin
from typing import List, Dict
try:
    from mpi4py import MPI
except ImportError:
    MPI_ON = False
else:
    MPI_ON = True


class SimulatorBase:
    """This is the parent class of all the simulators."""

    def __init__(self, cfg: Path = None) -> None:
        """Reads a yaml config file that specifies the chemical reactions and kinetic rates for the chemical system
        and sets key attributes.

        Parameters
        ----------
        cfg : str
            Path to config file that defines all chemical reactions and rates in yaml format.

        """

        self._config_path = cfg
        self._config_dictionary = None
        self._reactants = None
        self._products = None
        self._rates = None
        self._species = None
        self._propensity_species_ids = None
        self._reaction_matrix = None
        # dictionary to hold timings of various codeblocks for benchmarking
        self.timings: Dict[str, float] = {}

        if self._config_path:
            with open(self._config_path) as file:
                self._config_dictionary = yaml.load(file, Loader=yaml.Loader)
            self._set_rates()
            self._set_species_vector()
            self._set_reaction_matrix()

    def _set_rates(self) -> None:
        """create `self.rates` which is len(K) List[List[str, int]] where k'th element gives
        the name and value of the rate constant for the k'th reaction"""

        reactants = []
        products = []
        rates = []
        # need to use deepcopy because `self.update_rates()` will change rates in self._config_dictionary
        config_data = copy.deepcopy(self._config_dictionary)
        for reaction, rate in config_data['reactions'].items():
            reactants.append(reaction.replace(' ', '').split('->')[0].split('+'))
            products.append(reaction.replace(' ', '').split('->')[1].split('+'))
            rates.append(rate)

        self._reactants = reactants
        self._products = products
        self._rates = rates

    def _set_species_vector(self) -> None:
        """creates `self.species` which is len(N) List[str] where n'th element is the name of the n'th species"""

        species = []
        for reactants, products in zip(self._reactants, self._products):
            for molecule in reactants:
                species.append(molecule)
            for molecule in products:
                species.append(molecule)
        if '0' in species:
            species.remove('0')
        # remove duplicates and sort
        species = sorted(list(set(species)))

        self._species = species

    def _set_reaction_matrix(self) -> None:
        """has 2 jobs:
            - creates `self.reaction_matrix` which is shape(K,N) array where the [k,n] element
              gives the change in the n'th species for the k'th reaction
            - creates `self._propensenity_species_ids` which is len(K) List[List[int]] which gives the indices
              of `self.species` that are involved in the k'th reaction
        """

        n_reactions = len(self._reactants)
        n_species = len(self.species)
        reaction_matrix = np.zeros(shape=(n_reactions,n_species), dtype=int)
        for reaction, reactants, products in zip(reaction_matrix, self._reactants, self._products):
            for i, species in enumerate(self.species):
                # if this species in both a product and reactant, net effect is 0
                if species in reactants:
                    reaction[i] += -1
                if species in products:
                    reaction[i] += 1

        self._reaction_matrix = reaction_matrix
        self._propensity_species_ids = [[n for n in range(len(self._species)) if reaction_matrix[k,n] < 0]
                                        for k in range(len(reaction_matrix))]

    def get_propensity_strings(self) -> List[str]:
        """creates a readable list of the propensity of each reaction"""

        propensity_strings = []
        for propensity_ids, rate in zip(self._propensity_species_ids, self.rates):
            transition_rate = rate[0]
            for n in propensity_ids:
                transition_rate += f'c^{self._species[n]}'
            propensity_strings.append(transition_rate)

        return propensity_strings

    def run(self) -> None:
        """Runs the simulator of the child Class."""
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
    def rates(self) -> np.ndarray:
        return self._rates
    @rates.setter
    def rates(self, value):
        self._rates = value


class MasterEquation(SimulatorBase, CalculationsMixin, PlottingMixin):
    """Simulator of the Chemical Master Equationn."""

    def __init__(self, cfg: str = None, empty: bool = False, **initial_populations) -> None:
        """Uses the Chemical Master Equation (CME) to propagate the probability dynamics of a chemical system.

        The only input required by the user is the config file that specifies the elementary chemical reactions
        and kinet rates, and the initial nonzero population numbers of each species. The constructor will
        build the full constitutive state space in `self.constitutive_states` and the generator matrix in
        `self.generator_matrix`. To run the simulator, call the `self.run()` method with start, stop, and step
        arguments. The full P(t) trajectory will be stored in `self.results`.

        Parameters
        ----------
        cfg : str
            Path to yaml config file that specifies chemical reactions and kinetic rates
        **initial_populations : dict
            The initial population a particular species. If a species population is not specified,
            it's initial population is taken to be 0.

        """

        np.seterr(under='raise')
        super(MasterEquation, self).__init__(cfg)
        self._initial_populations = initial_populations
        if self._initial_populations:
            for species in self._initial_populations.keys():
                if species not in self._species:
                    raise KeyError(f'{species} is not a valid species. It must be in one of the '
                                   f'chemical reactions specified in the config file.')
        self._constitutive_states = None
        self._generator_matrix= None
        self._results = None
        self._dt = None
        self.comm = MPI.COMM_WORLD if MPI_ON else None
        self.size = self.comm.Get_size() if MPI_ON else None
        self.rank = self.comm.Get_rank() if MPI_ON else None
        self.parallel = MPI_ON and self.size > 1

        if self._initial_populations and not empty:
            with timeit() as set_constitutive_states:
                self._set_constitutive_states()
            with timeit() as set_generator_matrix:
                self._set_generator_matrix()
            self.timings['t_set_constitutive_states'] = set_constitutive_states.elapsed
            self.timings['t_set_generator_matrix'] = set_generator_matrix.elapsed

    def _set_constitutive_states(self) -> None:
        """Constructs all possible constitutive states from the intial state."""

        constitutive_states = []
        initial_state = [self._initial_populations[species] if species in self._initial_populations else 0
                         for species in self._species]
        constitutive_states.append(initial_state)

        # newly_added keeps track of the most recently accepted states
        newly_added_states = [np.array(initial_state)]
        while True:
            accepted_candidate_states = []
            for state in newly_added_states:
                # the idea here is that for each of the recently added states,
                # we iterate through each reaction to see if a transition is possible
                for reaction in self._reaction_matrix:
                    # gives a boolean array for which reactants are required
                    reactants_required = np.argwhere(reaction < 0).T
                    reactants_available = state > 0
                    # if this new state has all of the reactants available for the reaction
                    if np.all(reactants_available[reactants_required]):
                        # apply the reaction and add the new state into our list of constitutive
                        # states only if it is a new state that has not been previously visited
                        new_candidate_state = state + reaction
                        if list(new_candidate_state) not in constitutive_states:
                            accepted_candidate_states.append(new_candidate_state)
                            constitutive_states.append(list(new_candidate_state))
            # replace the old set of new states with these ones
            newly_added_states = accepted_candidate_states
            # once we reach the point where no new states are accessible we terminate
            if not newly_added_states:
                break

        self._constitutive_states = np.array(constitutive_states, dtype=np.int32)

    def get_readable_states(self) -> List:
        """creates a convenient human readable list of the constitutive states"""

        constitutive_states_strings = []
        for state in self._constitutive_states:
            word = []
            for population_number, species in zip(state, self.species):
                word.append(f'{population_number}{species}')
            constitutive_states_strings.append(word)

        return constitutive_states_strings

    def _set_generator_matrix(self) -> None:
        """Constructs the generator matrix.

        The generator matrix is an MxM matrix where M is the total number of states given
        by `len(self.constitutive_states)`.
        """

        M = len(self._constitutive_states)

        if self.parallel:
            start, stop = slice_tasks_for_parallel_workers(n_tasks=M, n_workers=self.size, rank=self.rank)
            local_blocksize = stop - start
            G_local = np.zeros(shape=(local_blocksize, M), dtype=np.float64)
            G_global = np.empty(shape=(M, M), dtype=np.float64)
        else:
            start, stop = 0, M
            G_local = G_global = np.zeros(shape=(M, M), dtype=np.float64)

        self._G_propensity_ids = {k: [] for k in range(len(self._reaction_matrix))}

        # we fill this matrix one row at a time
        for local_i, global_i in enumerate(ProgressBar(range(start, stop), position=self.rank,
                                           desc=f'rank {self.rank} working on generator matrix.')):
            # G_ij = propensity j -> i, so we are looking for any j state that can transition to i
            state_i = self._constitutive_states[global_i]
            for j in range(M):
                if global_i != j:
                    state_j = self._constitutive_states[j]
                    for k, reaction in enumerate(self._reaction_matrix):
                        # k'th reaction selected if a reaction exists
                        if np.array_equal(state_j + reaction, state_i):
                            # the indices of the species involved in the reaction
                            n_ids = self._propensity_species_ids[k]
                            # At the moment this assumes maximum stoichiometric coefficient of 1
                            combinatorial_factor = np.prod([state_j[n] for n in n_ids])
                            elementary_reaction_rate = self.rates[k][1]
                            reaction_propensity = combinatorial_factor * elementary_reaction_rate
                            G_local[local_i][j] = reaction_propensity
                            self._G_propensity_ids[k].append((global_i, j))
                            break

        #if self.parallel:
            # sendcounts tells comm.Allgatherv() how many elements are sent from each process
            #sendcounts = tuple(n_states * (slices[i].stop - slices[i].start) for i in range(self.size))
            # displacements tells comm.Allgatherv() the start index in the global array of each process' data
            #displacements = tuple(n_states * boundary.start for boundary in slices)
            #self.comm.Allgatherv(sendbuf=G_local,
            #                     recvbuf=(G_global, sendcounts, displacements, MPI.DOUBLE))
        for i in range(len(G_global)):
            # fix the diagonal to be the negative sum of the column
            G_global[i,i] = -np.sum(G_global[:, i])

        self._generator_matrix = G_global

    def get_readable_G(self) -> List[List[str]]:
        """creates a readable generator matrix with string names"""

        M = len(self.G)
        G_matrix_strings = [['0' for _ in range(M)] for _ in range(M)]
        propensity_strings = self.get_propensity_strings()
        for k in range(len(self.reaction_matrix)):
            for idx in self._G_propensity_ids[k]:
                i,j = idx
                G_matrix_strings[i][j] = propensity_strings[k]
        for j in range(M):
            diagonal = '-('
            for i in range(M):
                if G_matrix_strings[i][j] != '0':
                    diagonal += f'{G_matrix_strings[i][j]} + '
            G_matrix_strings[j][j] = diagonal[:-3] + ')'

        return G_matrix_strings

    def update_rates(self, **new_rates) -> None:
        """updates `self.rates` and `self.G` with new transition rates specified as keyword arguments"""

        for new_rate_string, new_rate_value in new_rates.items():
            for k, old_rate in enumerate(self._rates):
                if old_rate[0] == new_rate_string:
                    propensity_adjustment_factor = new_rate_value / old_rate[1]
                    # make sure to do this after saving the propensity factor
                    self._rates[k][1] = new_rate_value
                    # the generator matrix also changes when the rates change
                    G_elements_affected = self._G_propensity_ids[k]
                    for idx in G_elements_affected:
                        self._generator_matrix[idx] = self._generator_matrix[idx] * propensity_adjustment_factor
                    break
            else:
                raise KeyError(f'{new_rate_string} is not a valid rate for this system. The new rate'
                               f'must be one of the rates specified in the original config file.')
        # need to redo the diagonal elements as well
        for m in range(len(self._generator_matrix)):
            self._generator_matrix[m,m] = 0
            self._generator_matrix[m,m] = -np.sum(self._generator_matrix[:, m])

    def reset_rates(self) -> None:
        """resets `self.rates` to the original values from the config file"""

        rates_from_config = [rate for rate in self._config_dictionary['reactions'].values()]
        for k, (original_rate, current_rate) in enumerate(zip(rates_from_config, self._rates)):
            original_value = original_rate[1]
            current_value = current_rate[1]
            print(original_value, current_value)
            if original_value != current_value:
                propensity_adjustment_factor = original_value / current_value
                self._rates[k][1] = original_value
                G_elements_affected = self._G_propensity_ids[k]
                for idx in G_elements_affected:
                    self._generator_matrix[idx] = self._generator_matrix[idx] * propensity_adjustment_factor
        # need to redo the diagonal elements as well
        for m in range(len(self._generator_matrix)):
            self._generator_matrix[m,m] = 0
            self._generator_matrix[m,m] = -np.sum(self._generator_matrix[:, m])

    def run(self, start: float = None, stop: float = None, step: float = None) -> None:
        """Run."""

        self._dt = step
        # using np.round to avoid floating point precision errors
        n_timesteps = int(np.round((stop - start) / self._dt))
        M = len(self._constitutive_states)
        P = np.empty(shape=(n_timesteps, M), dtype=np.float64)
        P[0] = np.zeros(shape=M, dtype=np.float64)
        # fixing initial probability to be 1 in the intitial state
        P[0,0] = 1

        # only 1 process does this because naively parallelizing matrix*vector
        # operation is very slow compared to numpy optimized speeds
        if not self.parallel or self.rank == 0:
            with timeit() as matrix_exponential:
                Q = linalg.expm(self._generator_matrix*self._dt)
            with timeit() as run_time:
                for ts in ProgressBar(range(n_timesteps - 1), desc=f'rank {self.rank} running.'):
                    P[ts+1] = Q.dot(P[ts])
            self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
            self.timings['t_run'] = run_time.elapsed

        if self.parallel:
            self.comm.Bcast(P, root=0)

        self._results = P

    @property
    def states(self):
        return self._constitutive_states
    @states.setter
    def states(self, value):
        self._constitutive_states = value
    @property
    def G(self):
        return self._generator_matrix
    @property
    def P(self):
        return self._results
    @P.setter
    def P(self, value):
        self._results = value


class Gillespie(SimulatorBase):

    def __init__(self):
        super(Gillespie, self).__init__()

    def run(self):
        pass
