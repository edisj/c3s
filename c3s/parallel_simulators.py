import yaml
from yaml import CLoader
import numpy as np
from scipy.sparse import linalg
from pathlib import Path
from tqdm.autonotebook import tqdm
from .utils import timeit, slice_tasks_for_parallel_workers
from mpi4py import MPI


class BaseParallel:
    """Docstring TODO"""

    def __init__(self, cfg: str = 'reactions.cfg'):
        """Base class for all simulators.

        Reads a config file that specifies the chemical reactions and kinetic rates
        for the chemical system.

        Parameters
        ----------
        cfg : str
            Path to config file that defines all chemical reactions and rates.

        """

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.cfg_location = cfg
        self._species = None
        self._reaction_matrix = None
        self._rates_from_config = None
        self._rates = None
        self._rate_strings = None
        self._reactants = None
        self._products = None
        self.timings = {}

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

    def _load_data_from_yaml(self):
        """Reads the config file and stores the key:value pairs in the self._data dictionary."""

        data = {}
        config_file = Path.cwd() / self.cfg_location
        with open(config_file) as file:
            data.update(yaml.load(file, Loader=CLoader))

        return data

    def _process_data_from_config(self):
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

    def _set_species_vector(self):
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

    def _set_reaction_matrix(self):
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

    def _set_rates(self):
        """Constructs vector of kinetic rates."""

        rates_dictionary = self._rates_from_config
        rate_strings = [list(reaction.values())[0] for reaction in self._unprocessed_data['reactions']]
        rate_values = np.empty(shape=len(rate_strings))

        for i, rate in enumerate(rate_strings):
            rate_values[i] = rates_dictionary[rate]

        self._rates = rate_values
        self._rate_strings = rate_strings

    @property
    def species(self):
        return self._species

    @property
    def reaction_matrix(self):
        return self._reaction_matrix

    @property
    def rates(self):
        return self._rates

    @property
    def rate_strings(self):
        return self._rate_strings

    def run(self):
        pass


class MasterEquationParallel(BaseParallel):
    """Docstring TODO"""

    def __init__(self, initial_species=None, cfg='reactions.cfg'):
        """Docstring TODO.

        Parameters
        ----------
        initial_species : dict

        cfg : str
            Path to config file.

        """

        super(MasterEquationParallel, self).__init__(cfg)
        self.initial_state = None
        self.constitutive_states = None
        self.constitutive_states_strings = None
        self.generator_matrix= None
        self.generator_matrix_strings = None
        self.results = None

        self._initial_species = initial_species
        if initial_species is not None:
            with timeit() as set_initial_states:
                self._set_initial_state()
            with timeit() as set_constitutive_states:
                self._set_constitutive_states()
            with timeit() as set_generator_matrix:
                self._set_generator_matrices()

            self.timings['t_set_initial_states'] = set_initial_states.elapsed
            self.timings['t_set_constitutive_states'] = set_constitutive_states.elapsed
            self.timings['t_set_generator_matrix'] = set_generator_matrix.elapsed

    def _set_initial_state(self):
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

    def _set_generator_matrices(self):
        """Constructs the generator matrix."""

        N = len(self.constitutive_states)

        slices = make_balanced_slices(N, self.size, start=0, stop=N, step=1)
        # give each rank unique start and stop points
        start = slices[self.rank].start
        stop = slices[self.rank].stop
        bsize = stop - start
        # sendcounts is used for Gatherv() to know how many elements are sent from each rank
        sendcounts = N * np.array([slices[i].stop - slices[i].start for i in range(self.size)])

        G_global_buffer = np.empty(shape=(N,N), dtype=np.int32)
        G_local_buffer = np.empty(shape=(bsize, N), dtype=np.int32)

        # now each process has unique scanning range
        for index, i in enumerate(range(start, stop)):
            #new_row = []
            state_i = self.constitutive_states[i]
            for j in range(N):
                state_j = self.constitutive_states[j]
                if i == j:
                    #rate_string = '-'
                    rate_value = 0
                for k, reaction in enumerate(self.reaction_matrix):
                    if list(state_i + reaction) == state_j:
                        #rate_string = self.rate_strings[k]
                        rate_value = self.rates[k]
                        break
                else:
                    #rate_string = '0'
                    rate_value = 0
                #new_row.append(rate_string)
                G_local_buffer[index][j] = rate_value

            G_local_buffer[index][i] = -np.sum(G_local_buffer[index])

        self.comm.Gatherv(sendbuf=G_local_buffer, recvbuf=(G_global_buffer, sendcounts), root=0)
        self.comm.Bcast(G_global_buffer, root=0)
        self.generator_matrix = G_global_buffer
        #self.generator_matrix_strings = generator_matrix_strings

    def update_rates(self, new_rates: dict):
        """Updates rates.

        Parameters
        ----------
        new_rates : dict


        """

        for key, value in new_rates.items():
            self._rates_from_config[key] = value
        self._set_rates()
        self._set_generator_matrices()

    def reset_rates(self):
        """Resets rates to values from the config file."""

        self._rates_from_config = self._unprocessed_data['rates']
        self._set_rates()
        self._set_generator_matrices()

    def run(self, start=None, stop=None, step=None):
        """Run.

        Parameters
        ----------
        start :
            Set whether or not to display who the quote is from.
        stop :
            Set whether or not to display who the quote is from.
        step :
            Set whether or not to display who the quote is from.

        """

        n_steps = int((stop-start) / step)
        dt = step

        N = len(self.constitutive_states)
        P_0 = np.zeros(shape=N)
        P_t = np.empty(shape=(n_steps,N))

        for i, state in enumerate(self.constitutive_states):
            if np.array_equal(state, self.initial_state):
                P_0[i] = 1
                break

        P_t[0] = P_0

        with timeit() as G_times_dt:
            Gdt = self.generator_matrix * dt
        with timeit() as matrix_exponential:
            propagator = linalg.expm(Gdt)

        with timeit() as run_time:
            for i in tqdm(range(n_steps - 1), leave=True, desc=f'running on process {self.rank}'):
                P_t[i+1] = P_t[i].dot(propagator)

        self.timings['t_G_times_dt'] = G_times_dt.elapsed
        self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
        self.timings['t_run'] = run_time.elapsed

        self.results = P_t


class GillespieParallel(BaseParallel):

    def __init__(self):
        super(GillespieParallel, self).__init__()

    def run(self):
        pass
