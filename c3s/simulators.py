import numpy as np
import yaml
from scipy.sparse import linalg
from pathlib import Path
from .utils import timeit, ProgressBar, slice_tasks_for_parallel_workers
from typing import List, Dict
try:
    from mpi4py import MPI
except ImportError:
    MPI_ON = False
else:
    MPI_ON = True


class Base:
    """Docstring TODO"""

    def __init__(self, cfg: str) -> None:
        """Base class for all simulators.

        Reads a config file that specifies the chemical reactions and kinetic rates
        for the chemical system.

        Parameters
        ----------
        cfg : str
            Path to config file that defines all chemical reactions and rates.

        """

        self.cfg_location = cfg
        self._species = None
        self._reaction_matrix = None
        self._rates_from_config = None
        self._rates = None
        self._rate_strings = None
        self._reactants = None
        self._products = None
        # dictionary to hold timings of various codeblocks for benchmarking
        self.timings: Dict[str, float] = {}

        self._unprocessed_data = self._load_data_from_yaml()

        with timeit() as process_data_from_config:
            self._process_data_from_config()
        with timeit() as set_species:
            self._set_species_vector()
        with timeit() as set_reaction_matrix:
            self._set_reaction_matrix()
        with timeit() as set_rates:
            self._set_rates()

        self.timings['t_process_data_from_config'] = process_data_from_config.elapsed
        self.timings['t_set_species_vector'] = set_species.elapsed
        self.timings['t_set_reaction_matrix'] = set_reaction_matrix.elapsed
        self.timings['t_set_rates'] = set_rates.elapsed

    def _load_data_from_yaml(self) -> Dict:
        """Reads the config file and stores the key:value pairs in the self._unprocessed_data dictionary."""

        unprocessed_data = {}
        config_file = Path.cwd() / self.cfg_location
        with open(config_file) as file:
            unprocessed_data.update(yaml.load(file, Loader=yaml.CLoader))

        return unprocessed_data

    def _process_data_from_config(self) -> None:
        """Reads the data from config file to get reactants, products, and rates."""

        reactions = [list(reaction.keys())[0] for reaction in self._unprocessed_data['reactions']]
        reactions = [reaction.replace(' ', '') for reaction in reactions]
        reactants = [reaction.split("->")[0] for reaction in reactions]
        reactants = [reactant.split('+') for reactant in reactants]
        products = [reaction.split("->")[1] for reaction in reactions]
        products = [product.split('+') for product in products]

        self._reactants = reactants
        self._products = products
        self._rates_from_config = self._unprocessed_data['rates'].copy()

    def _set_species_vector(self) -> None:
        """Constructs vector of strings that specify the set of unique molecular species in the system."""

        reactants = self._reactants
        products = self._products

        all_species = []
        for reactant, product in zip(reactants, products):
            for species in reactant:
                all_species.append(species)
            for species in product:
                all_species.append(species)

        if '0' in all_species:
            all_species.remove('0')
        # remove duplicates and sort
        all_species = sorted(list(set(all_species)))

        self._species = all_species

    def _set_reaction_matrix(self) -> None:
        """Constructs the stochiometric reaction matrix."""

        reactants = self._reactants
        products = self._products

        n_reactions = len([list(reaction.keys())[0] for reaction in self._unprocessed_data['reactions']])
        n_species = len(self.species)

        reaction_matrix = np.zeros(shape=(n_reactions,n_species), dtype=int)

        for row, reactant, product in zip(reaction_matrix, reactants, products):
            for i, species in enumerate(self.species):
                if species in reactant:
                    row[i] += -1
                if species in product:
                    row[i] += 1

        self._reaction_matrix = reaction_matrix

    def _set_rates(self) -> None:
        """Constructs vector of kinetic rates."""

        rates_dictionary = self._rates_from_config
        rate_strings = [list(reaction.values())[0] for reaction in self._unprocessed_data['reactions']]
        rate_values = np.empty(shape=len(rate_strings))

        for i, rate in enumerate(rate_strings):
            rate_values[i] = rates_dictionary[rate]

        self._rates = rate_values
        self._rate_strings = rate_strings

    @property
    def species(self) -> List[str]:
        return self._species

    @property
    def reaction_matrix(self) -> np.ndarray:
        return self._reaction_matrix

    @property
    def rates(self) -> np.ndarray:
        return self._rates

    @property
    def rate_strings(self) -> List[str]:
        return self._rate_strings

    def run(self) -> None:
        """Runs the simulator of the child Class."""
        pass


class MasterEquation(Base):
    """Docstring TODO"""

    def __init__(self, cfg: str, initial_species: Dict = None) -> None:
        """Docstring TODO.

        Parameters
        ----------
        initial_species : dict
        cfg : str
            Path to config file that specifies chemical reactions and kinetic rates.

        """

        super(MasterEquation, self).__init__(cfg)
        self.initial_state = None
        self.constitutive_states = None
        self.constitutive_states_strings = None
        self.generator_matrix= None
        self.results = None
        self._initial_species = initial_species
        # MPI communicator
        self.comm = MPI.COMM_WORLD if MPI_ON else None
        # unique process id
        self.rank = self.comm.Get_rank() if MPI_ON else None
        # total number of processes available
        self.size = self.comm.Get_size() if MPI_ON else None
        self.parallel = MPI_ON and self.size > 1

        if initial_species is not None:
            with timeit() as set_initial_states:
                self._set_initial_state()
            with timeit() as set_constitutive_states:
                self._set_constitutive_states()
            with timeit() as set_generator_matrix:
                self._set_generator_matrix()

            self.timings['t_set_initial_states'] = set_initial_states.elapsed
            self.timings['t_set_constitutive_states'] = set_constitutive_states.elapsed
            self.timings['t_set_generator_matrix'] = set_generator_matrix.elapsed

    def _set_initial_state(self) -> None:
        """Sets the initial state vector."""

        initial_state = np.zeros(shape=(len(self.species)), dtype=np.int32)
        all_species = np.array(self.species)

        for species, quantity in self._initial_species.items():
            if species in self.species:
                i = np.argwhere(all_species == species)
                initial_state[i] = quantity

        self.initial_state = initial_state

    def _set_constitutive_states(self):
        """Constructs all possible constitutive states from the intial state."""

        constitutive_states = [list(self.initial_state)]
        newly_added_unique_states = [self.initial_state]
        while True:
            accepted_candidate_states = []
            for state in newly_added_unique_states:
                for reaction in self.reaction_matrix:
                    reactants_required = reaction < 0
                    indices = np.argwhere(reactants_required).transpose()
                    reactants_available = state > 0
                    if np.all(reactants_available[indices]):
                        new_candidate_state = state + reaction
                        if list(new_candidate_state) not in constitutive_states:
                            accepted_candidate_states.append(new_candidate_state)
                            constitutive_states.append(list(new_candidate_state))

            newly_added_unique_states = [state for state in accepted_candidate_states]
            if not newly_added_unique_states:
                break

        constitutive_states_strings = []
        for state in constitutive_states:
            word = []
            for quantity, species in zip(state, self.species):
                word.append(f'{quantity}{species}')

            constitutive_states_strings.append(word)

        self.constitutive_states = constitutive_states
        self.constitutive_states_strings =  constitutive_states_strings

    def _set_generator_matrix(self) -> None:
        """Constructs the generator matrix."""

        n_states = len(self.constitutive_states)

        if self.parallel:
            slices = slice_tasks_for_parallel_workers(n_tasks = n_states, n_workers = self.size)
            # sendcounts tells comm.Allgatherv() how many elements are sent from each process
            sendcounts = tuple(n_states*(slices[i].stop - slices[i].start) for i in range(self.size))
            # displacements tells comm.Allgatherv() the start index in the global array of each process' data
            displacements = tuple(n_states*boundary.start for boundary in slices)
            start = slices[self.rank].start
            stop = slices[self.rank].stop
            local_blocksize = stop - start
            # local buffer that only this process sees
            generator_matrix_local = np.zeros(shape=(local_blocksize, n_states), dtype=np.int32)
            # buffer that each process will fill in by receiving data from every other process
            generator_matrix_global = np.empty(shape=(n_states, n_states), dtype=np.int32)
        else:
            start = 0
            stop = n_states
            generator_matrix_local = generator_matrix_global = np.zeros(shape=(n_states, n_states), dtype=np.int32)

        for local_i, i in enumerate(ProgressBar(range(start, stop), position=self.rank,
                                                desc=f'rank {self.rank} working on generator matrix.')):
            state_i = self.constitutive_states[i]
            for j in range(n_states):
                state_j = self.constitutive_states[j]
                if i != j:
                    for k, reaction in enumerate(self.reaction_matrix):
                        if list(state_i + reaction) == state_j:
                            generator_matrix_local[local_i][j]  = self.rates[k]
                            break
            # fix diagonal elements after completing the row
            generator_matrix_local[local_i][i] = -np.sum(generator_matrix_local[local_i])

        if self.parallel:
            self.comm.Allgatherv(sendbuf=generator_matrix_local,
                                 recvbuf=(generator_matrix_global, sendcounts, displacements, MPI.INT))

        self.generator_matrix = generator_matrix_global

    def update_rates(self, new_rates: Dict[str, float]) -> None:
        """Updates rates."""

        for key, value in new_rates.items():
            self._rates_from_config[key] = value
        self._set_rates()
        self._set_generator_matrix()

    def reset_rates(self) -> None:
        """Resets rates to values from the config file."""

        self._rates_from_config = self._unprocessed_data['rates']
        self._set_rates()
        self._set_generator_matrix()

    def run(self, initial_time: float = None, final_time: float = None, dt: float = None, ONE_PROC=False) -> None:
        """Run."""

        # using np.round to avoid floating point precision errors
        n_timesteps = int(np.round((final_time - initial_time) / dt))
        n_states = len(self.constitutive_states)
        probability_vectors = np.empty(shape=(n_timesteps, n_states))

        if ONE_PROC:
            # only 1 process does this
            if not self.parallel or self.rank == 0:
                initial_probability_vector = np.zeros(shape=n_states)
                # set the initial probability vector to have probability 1 at the initial state
                for i, state in enumerate(self.constitutive_states):
                    if np.array_equal(state, self.initial_state):
                        initial_probability_vector[i] = 1
                        break
                probability_vectors[0] = initial_probability_vector

                with timeit() as matrix_exponential:
                    # propagator matrix
                    Q = linalg.expm(self.generator_matrix*dt)
                with timeit() as run_time:
                    for ts in ProgressBar(range(n_timesteps - 1), desc=f'rank {self.rank} running.'):
                        probability_vectors[ts+1] = probability_vectors[ts].dot(Q)

                self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
                self.timings['t_run'] = run_time.elapsed
            else:
                self.timings['t_matrix_exponential'] = 0.0
                self.timings['t_run'] = 0.0
        else:
            initial_probability_vector = np.zeros(shape=n_states)
            # set the initial probability vector to have probability 1 at the initial state
            for i, state in enumerate(self.constitutive_states):
                if np.array_equal(state, self.initial_state):
                    initial_probability_vector[i] = 1
                    break
            probability_vectors[0] = initial_probability_vector

            with timeit() as matrix_exponential:
                # propagator matrix
                Q = linalg.expm(self.generator_matrix * dt)
            with timeit() as run_time:
                for ts in ProgressBar(range(n_timesteps - 1), desc=f'rank {self.rank} running.'):
                    probability_vectors[ts + 1] = probability_vectors[ts].dot(Q)

            self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
            self.timings['t_run'] = run_time.elapsed

        if self.parallel:
            self.comm.Bcast(probability_vectors, root=0)

        self.results = probability_vectors


class Gillespie(Base):

    def __init__(self):
        super(Base, self).__init__()

    def run(self):
        pass
