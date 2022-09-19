import math
import yaml
import copy
import h5py
import random
import numpy as np
from collections import namedtuple
from scipy.sparse.linalg import expm
from c3s.utils import timeit, ProgressBar, slice_tasks_for_parallel_workers
from typing import List, Dict
try:
    from mpi4py import MPI
except ImportError:
    MPI_ON = False
else:
    MPI_ON = True


class SimulatorBase:
    """This is the parent class of all the simulators."""

    np.seterr(under='raise', over='raise')

    def __init__(self, cfg, filename=None, mode='x', system_name=None,
                 initial_state=None, initial_populations=None, max_populations=None, empty=False):
        """Reads a yaml config file that specifies the chemical reactions and kinetic rates for the chemical system
        and sets important attributes.

        Parameters
        ----------
        cfg : str
            Path to config file that defines all chemical reactions and rates in yaml format.

        """

        self._config_path = cfg
        self._config_dictionary = None
        self._file = None
        self._initial_populations = initial_populations
        self._initial_state = initial_state
        if self._initial_state and self._initial_populations:
            raise ValueError("Do not specify both the `initial_state` and `initial_populations` parameters. "
                             "Use one or the other.")
        self._max_populations = max_populations
        self._empty = empty
        # set up the MPI communicator and assign rank ids to each process
        if MPI_ON:
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        else:
            self.comm, self.size, self.rank = None, None, None
        self.parallel = MPI_ON and self.size > 1
        self._reactants = None
        self._products = None
        self._rates = None
        self._species = None
        self._propensity_indices = None
        self._reaction_matrix = None
        self._results = None
        self._dt = None
        # dictionary to hold timings of various codeblocks for benchmarking
        self.timings: Dict[str, float] = {}

        # now that the setup is taken care of, begin filling the data structures
        if self._config_path:
            with open(self._config_path) as file:
                self._config_dictionary = yaml.load(file, Loader=yaml.Loader)
            self._set_rates()
            self._set_species_vector()
            self._set_initial_state()
            self._set_reaction_matrix_and_propensity_indices()
        # open filestream handle under `self._file` if a `filename` was provided as parameter
        if filename:
            if system_name is None:
                # default system name uses initial species counts
                self._system_name = ''
                for species, count in self._initial_populations.items():
                    self._system_name += f'{count}{species}'
            self._open_HDF5_file(filename, mode)
            self._create_HDF5_group(self._system_name)

    def _open_HDF5_file(self, filename, mode):
        """"""
        if self.parallel:
            self._file = h5py.File(filename, mode=mode, driver='mpio', comm=self.comm)
        else:
            self._file = h5py.File(filename, mode=mode)

    def _create_HDF5_group(self, name):
        return self._file.require_group(name)

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
            for molecule in reactants:
                #if molecule == '0':
                #    if not self._max_populations:
                #        raise ValueError()
                #    birth_molecule = reactants[k]
                #    if birth_mo
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

        n_reactions = len(self._reactants)
        n_species = len(self.species)
        reaction_matrix = np.zeros(shape=(n_reactions,n_species), dtype=np.int32)
        for reaction, reactants, products in zip(reaction_matrix, self._reactants, self._products):
            for i, species in enumerate(self.species):
                # if this species in both a product and reactant, net effect is 0
                if species in reactants:
                    reaction[i] += -1
                if species in products:
                    reaction[i] += 1

        self._reaction_matrix = reaction_matrix
        self._propensity_indices = [[n for n in range(len(self._species)) if reaction_matrix[k,n] < 0]
                                     for k in range(len(reaction_matrix))]

    def get_propensity_strings(self):
        """creates a readable list of the propensity of each reaction"""

        propensity_strings: List[str] = []
        for propensity_ids, rate in zip(self._propensity_indices, self.rates):
            transition_rate = rate[0]
            for n in propensity_ids:
                transition_rate += f'c^{self._species[n]}'
            propensity_strings.append(transition_rate)

        return propensity_strings

    def run(self, *args, **kwargs):
        """runs the simulator of the child class"""
        pass

    def reset_rates(self):
        """resets `self.rates` to the original values from the config file"""

        rates_from_config = [rate for rate in self._config_dictionary['reactions'].values()]
        original_rates = {rate[0]: rate[1] for rate in rates_from_config}
        self.update_rates(original_rates)

    def update_rates(self, new_rates):
        """the child class decides what to do with this"""
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


class Gillespie(SimulatorBase):
    """The Gillespie stochastic simulation algorithm (SSA)."""

    def __init__(self, cfg, filename=None, mode='x', system_name=None,
                 initial_state=None, initial_populations=None, max_populations=None, empty=False):
        """Uses Gillespie's stochastic simulation algorithm to generate a trajectory of a random walker that is
        defined by a molecular population vector.

        Parameters
        ----------
        cfg : str
            Path to yaml config file that specifies chemical reactions and kinetic rates
        filename : str
        mode : str, default='x'
        system_name : str
        initial_state : list of int or array_like
        initial_populations : dict, default=None
            The initial population a particular species. If a species population is
            not specified, it's initial population is taken to be 0.
        max_populations : dict
        empty : bool

        """

        super(Gillespie, self).__init__(cfg=cfg, filename=filename, mode=mode, system_name=system_name,
                                        initial_state=initial_state, initial_populations=initial_populations,
                                        max_populations=max_populations, empty=empty)

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
              call the section that the i'th event takes up on the line L_i
            - sample a uniform random number, u,  between 0 and 1
            - find the smallest index k such that u < L_1 + ... + L_k + ... + L_K
            - that index, k, gives which event was selected

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

    def run(self, N_timesteps, overwrite=False):
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
        Trajectory = namedtuple("Trajectory", ["states", "holding_times"])
        sequence_of_states = np.empty(shape=(N_timesteps, N_species), dtype=np.int32)
        holding_times = np.empty(shape=N_timesteps-1, dtype=np.float64)
        trajectory = Trajectory(sequence_of_states, holding_times)

        currState = np.array(self._initial_state, dtype=np.int32)
        sequence_of_states[0] = currState
        for ts in range(N_timesteps - 1):
            propensity_vector = self._get_propensity_vector(currState)
            nextState = self._get_next_state(currState, propensity_vector)
            holding_time = self._sample_holding_time(propensity_vector)
            trajectory.states[ts+1] = nextState
            trajectory.holding_times[ts] = holding_time
            # set current state to the new state and proceed along the journey
            currState = nextState

        self._results = trajectory

    def run_many_iterations(self, N_iterations, N_timesteps):
        """"""

        N_species = len(self._species)
        if self.parallel:
            start, stop = slice_tasks_for_parallel_workers(n_tasks=N_iterations, n_workers=self.size, rank=self.rank)
            local_blocksize = stop - start
            trajectories = np.empty(shape=(local_blocksize, N_timesteps, N_species), dtype=np.int32)
            holding_times = np.empty(shape=(local_blocksize, N_timesteps - 1), dtype=np.float64)
            trajectories_global = np.empty(shape=(N_iterations, N_timesteps, N_species), dtype=np.int32)
            holding_times_global = np.empty(shape=(N_iterations, N_timesteps - 1), dtype=np.float64)
        else:
            start, stop = 0, N_iterations
            trajectories = trajectories_global = np.empty(shape=(N_iterations, N_timesteps, N_species), dtype=np.int32)
            holding_times = holding_times_global = np.empty(shape=(N_iterations, N_timesteps - 1), dtype=np.float64)

        for ts in range(start, stop):
            self.run(N_timesteps, overwrite=True)
            trajectories[ts] = self.trajectory.states
            holding_times[ts] = self.trajectory.holding_times

        if self.parallel:
            traj_sendcounts = self.comm.allgather(trajectories.size)
            traj_displacements = self.comm.allgather(self.rank * trajectories.size)
            h_sendcounts = self.comm.allgather(holding_times.size)
            h_displacements = self.comm.allgather(self.rank * holding_times.size)
            self.comm.Allgatherv(
                sendbuf=trajectories, recvbuf=(trajectories_global, traj_sendcounts, traj_displacements, MPI.INT))
            self.comm.Allgatherv(
                sendbuf=holding_times, recvbuf=(holding_times_global, h_sendcounts, h_displacements, MPI.DOUBLE))

        self._results = (trajectories_global, holding_times_global)

    @property
    def trajectory(self):
        return self._results
    @trajectory.setter
    def trajectory(self, value):
        self._results = value


class ChemicalMasterEquation(SimulatorBase):
    """Simulator class of the Chemical Master Equationn (CME)."""

    def __init__(self, cfg, filename=None, mode='x', system_name=None,
                 initial_state=None, initial_populations=None, max_populations=None, empty=False):
        """Uses the Chemical Master Equation to propagate the time
           evolution of the probability dynamics of a chemical system.

        The only input required by the user is the config file that specifies the elementary chemical reactions
        and kinet rates, and the initial nonzero population numbers of each species. The constructor will
        build the full constitutive state space in `self.constitutive_states` and the generator matrix in
        `self.generator_matrix`. To run the simulator, call the `self.run()` method with start, stop, and step
        arguments. The full P(t) trajectory will be stored in `self.results`.

        Parameters
        ----------
        cfg : str
            Path to yaml config file that specifies chemical reactions and kinetic rates
        filename : str
        mode : str, default='x'
        system_name : str
        initial_state : list of int or array_like
        initial_populations : dict, default=None
            The initial population a particular species. If a species population is not specified,
            it's initial population is taken to be 0.
        max_populations : dict
        empty : bool

        """
        super(ChemicalMasterEquation, self).__init__(cfg, filename=filename, mode=mode, system_name=system_name,
                                                     initial_state=initial_state, initial_populations=initial_populations,
                                                     max_populations=max_populations, empty=empty)
        self._constitutive_states = None
        self._generator_matrix= None

        if not empty:
            with timeit() as set_constitutive_states:
                self._set_constitutive_states()
            with timeit() as set_generator_matrix:
                self._set_generator_matrix()
            self.timings['t_set_constitutive_states'] = set_constitutive_states.elapsed
            self.timings['t_set_generator_matrix'] = set_generator_matrix.elapsed

            if self._file:
                self._file[self._system_name].attrs['M'] = len(self._constitutive_states)

    def _set_constitutive_states(self):
        """Constructs all possible constitutive states from the intial state."""

        constitutive_states = [self._initial_state]

        # newly_added keeps track of the most recently accepted states
        newly_added_states = [np.array(self._initial_state)]
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

    def get_readable_states(self):
        """creates a convenient human readable list of the constitutive states"""

        constitutive_states_strings: List[List[str]] = []
        for state in self._constitutive_states:
            word = []
            for population_number, species in zip(state, self.species):
                word.append(f'{population_number}{species}')
            constitutive_states_strings.append(word)

        return constitutive_states_strings

    def _set_generator_matrix(self):
        """Constructs the generator matrix.

        The generator matrix is an MxM matrix where M is the total number of states given
        by `len(self.constitutive_states)`.
        """

        M = len(self._constitutive_states)
        K = len(self._reaction_matrix)
        if self.parallel:
            start, stop = slice_tasks_for_parallel_workers(n_tasks=M, n_workers=self.size, rank=self.rank)
            local_blocksize = stop - start
            G_local = np.zeros(shape=(local_blocksize, M), dtype=np.float64)
            G = np.zeros(shape=(M, M), dtype=np.float64)
        else:
            start, stop = 0, M
            G = G_local = np.zeros(shape=(M, M), dtype=np.float64)

        self._G_propensity_ids = {k: [] for k in range(K)}

        # local_i = global_i in the serial case
        for local_i, global_i in enumerate(ProgressBar(range(start, stop),
                                                       position=self.rank,
                                                       desc=f'rank {self.rank} working on generator matrix...')):
            # G_ij = propensity j -> i, so we are looking for any j state that can transition to i
            state_i = self._constitutive_states[global_i]
            for j in range(M):
                if global_i != j:
                    state_j = self._constitutive_states[j]
                    for k, reaction in enumerate(self._reaction_matrix):
                        # k'th reaction selected if a reaction exists
                        if np.array_equal(state_j + reaction, state_i):
                            # the indices of the species involved in the reaction
                            n_ids = self._propensity_indices[k]
                            # h is the combinatorial factor for number of reactions attempting to fire
                            # At the moment this assumes maximum stoichiometric coefficient of 1
                            h = np.prod([state_j[n] for n in n_ids])
                            # lambda_ is the elementary reaction rate for the k'th reaction
                            lambda_ = self._rates[k][1]
                            reaction_propensity = h*lambda_
                            G_local[local_i][j] = reaction_propensity
                            self._G_propensity_ids[k].append((global_i, j))
                            break

        if self.parallel:
            G_sendcounts = self.comm.allgather(M * local_blocksize)
            G_displacements = self.comm.allgather(M * start)
            self.comm.Allgatherv(sendbuf=G_local,
                                 recvbuf=(G, G_sendcounts, G_displacements, MPI.DOUBLE))
            # each process also has to communicate which indices it collected for its local block
            for k in range(K):
                indices_collected_by_this_process = np.array(self._G_propensity_ids[k], dtype=np.int32)
                propensity_id_sendcounts = self.comm.allgather(indices_collected_by_this_process.size)
                propensity_id_displacements = [count-2 for count in propensity_id_sendcounts]
                recvbuf = np.zeros(shape=(int(sum(propensity_id_sendcounts)/2), 2), dtype=np.int32)
                self.comm.Allgatherv(sendbuf=indices_collected_by_this_process,
                                     recvbuf=(recvbuf, propensity_id_sendcounts, propensity_id_displacements, MPI.INT))
                self._G_propensity_ids[k] = [tuple(index) for index in recvbuf.tolist()]

        for i in range(M):
            # fix the diagonal to be the negative sum of the column
            G[i, i] = -np.sum(G[:, i])

        self._generator_matrix = G

    def get_readable_G(self):
        """creates a readable generator matrix with string names"""

        M = len(self.G)
        readable_G = [['0' for _ in range(M)] for _ in range(M)]
        propensity_strings = self.get_propensity_strings()
        for k in range(len(self.reaction_matrix)):
            for idx in self._G_propensity_ids[k]:
                i,j = idx
                readable_G[i][j] = propensity_strings[k]
        for j in range(M):
            diagonal = '-('
            for i in range(M):
                if readable_G[i][j] != '0':
                    diagonal += f'{readable_G[i][j]} + '
            readable_G[j][j] = diagonal[:-3] + ')'

        return readable_G

    def update_rates(self, new_rates):
        """updates `self.rates` and `self.G` with new transition rates"""

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

    def run(self, start, stop, step, overwrite=False):
        """Runs the chemical master equation simulation.

        Parameters
        ----------
        start : int or float
        stop  : int or float
        step  : int or float

        """

        if self._results is not None and not overwrite:
            raise ValueError("Data from previous run found in `self.P_trajectory`. "
                             "To write over this data, set the `overwrite=True`")
        self._dt = step
        # using np.round to avoid floating point precision errors
        n_timesteps = int(np.round((stop - start) / self._dt))
        M = len(self._constitutive_states)
        P_trajectory = np.empty(shape=(n_timesteps, M), dtype=np.float64)
        # fixing initial probability to be 1 in the intitial state
        P_trajectory[0] = np.zeros(shape=M, dtype=np.float64)
        P_trajectory[0, 0] = 1

        # only 1 process does this because naively parallelizing matrix*vector
        # operation is very slow compared to numpy optimized speeds
        if not self.parallel or self.rank == 0:
            with timeit() as matrix_exponential:
                Q = expm(self._generator_matrix * self._dt)
            with timeit() as run_time:
                for ts in ProgressBar(range(n_timesteps - 1), desc=f'rank {self.rank} running.'):
                    P_trajectory[ts + 1] = Q.dot(P_trajectory[ts])
            self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
            self.timings['t_run'] = run_time.elapsed

        if self.parallel:
            self.comm.Bcast(P_trajectory, root=0)

        self._results = P_trajectory

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
    def P_trajectory(self):
        return self._results
    @P_trajectory.setter
    def P_trajectory(self, value):
        self._results = value

    def calculate_mutual_information(self, X, Y, timestep='all', base=2, run_name=None):
        """Calculates the mutual information."""

        log = math.log2 if base == 2 else math.log

        if timestep != 'all':
            raise ValueError("other timestep values not implemented yet..")
        try:
            n_timesteps = len(self.P_trajectory)
        except TypeError:
            raise ValueError('No data found in `self.P_trajectory`.')

        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            Y = [Y]
        X = sorted(X)
        Y = sorted(Y)
        if X+Y != sorted(X+Y):
            X, Y = Y, X

        x_map = self._get_point_mappings(X)
        y_map = self._get_point_mappings(Y)
        xy_map = self._get_point_mappings(X + Y)

        if self.parallel:
            start, stop = slice_tasks_for_parallel_workers(n_tasks=n_timesteps, n_workers=self.size, rank=self.rank)
            local_blocksize = stop - start
            mutual_information_local = np.empty(shape=local_blocksize, dtype=np.float64)
            mutual_information_global = np.empty(shape=n_timesteps, dtype=np.float64)
        else:
            start, stop = 0, n_timesteps
            local_blocksize = stop - start
            mutual_information_local = mutual_information_global = np.empty(shape=local_blocksize, dtype=np.float64)

        with timeit() as calculation_block:
            for i, ts in enumerate(ProgressBar(range(start,stop), position=self.rank,
                                               desc=f'rank {self.rank} calculating mutual information.')):
                P = self.P_trajectory[ts]
                mutual_information = 0
                for x_point in x_map:
                    for y_point in y_map:
                        xy_point = x_point + y_point
                        if xy_point not in xy_map:
                            # skip cases where the concatenated coordinate tuples
                            # were never in the joint distribution to begin with
                            continue
                        idx = x_map[x_point]
                        idy = y_map[y_point]
                        idxy = xy_map[xy_point]
                        p_xy = np.sum(P[idxy])
                        if p_xy == 0:
                            # add zero to the sum if p_xy is 0
                            # need to do this because 0*np.log(0) returns an error
                            continue
                        p_x = np.sum(P[idx])
                        p_y = np.sum(P[idy])
                        this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
                        mutual_information += this_term
                mutual_information_local[i] = mutual_information
        self.timings['t_calculate_mutual_information'] = calculation_block.elapsed

        if self.parallel:
            sendcounts = self.comm.allgather(local_blocksize)
            displacements = self.comm.allgather(start)
            self.comm.Allgatherv(sendbuf=mutual_information_local,
                                 recvbuf=(mutual_information_global, sendcounts, displacements, MPI.DOUBLE))
        if self._file:
            HDF5_group = self._file[self._system_name]
            run_name = 'run_' + str(len(self._file[self._system_name])) if run_name is None else run_name
            mi_dataset = HDF5_group.create_dataset(f'{run_name}/mutual_information', data=mutual_information_global)
            mi_dataset.attrs['rates'] = [rate[1] for rate in self._rates]
            mi_dataset.attrs['dt'] = self._dt

        return mutual_information_global

    def calculate_instantaneous_mutual_information(self, X, Y, timestep='all', base=2, run_name=None):

        log = math.log2 if base == 2 else math.log

        if timestep != 'all':
            raise ValueError("other timestep values not implemented yet..")
        try:
            n_timesteps = len(self.P_trajectory)
        except TypeError:
            raise ValueError('No data found in `self.P_trajectory`.')

        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            Y = [Y]
        X = sorted(X)
        Y = sorted(Y)
        if X + Y != sorted(X + Y):
            X, Y = Y, X

        Deltas_x = self._get_Delta_vectors(X)
        Deltas_y = self._get_Delta_vectors(Y)
        Deltas_xy = self._get_Delta_vectors(X + Y)

        if self.parallel:
            start, stop = slice_tasks_for_parallel_workers(n_tasks=n_timesteps, n_workers=self.size, rank=self.rank)
            local_blocksize = stop - start
            mutual_information_local = np.empty(shape=local_blocksize, dtype=np.float64)
            mutual_information_global = np.empty(shape=n_timesteps, dtype=np.float64)
        else:
            start, stop = 0, n_timesteps
            local_blocksize = stop - start
            mutual_information_local = mutual_information_global = np.empty(shape=local_blocksize, dtype=np.float64)

        with timeit() as calculation_block:
            for i, ts in enumerate(ProgressBar(range(start, stop), position=self.rank,
                                               desc=f'rank {self.rank} calculating mutual information.')):
                P = self.P_trajectory[ts]
                mutual_information = 0
                for x, Dx in Deltas_x.items():
                    for y, Dy in Deltas_y.items():
                        xy = x + y
                        if xy not in Deltas_xy:
                            # skip cases where the concatenated coordinate tuples
                            # were never in the joint distribution to begin with
                            continue
                        Dxy = Deltas_xy[xy]
                        p_xy = np.dot(P, Dxy)
                        if p_xy == 0:
                            # add zero to the sum if p_xy is 0
                            # need to do this because 0*np.log(0) returns an error
                            continue
                        p_x = np.dot(P, Dx)
                        p_y = np.dot(P, Dy)
                        this_term = p_xy * log(p_xy / (p_x * p_y))
                        mutual_information += this_term

                mutual_information_local[i] = mutual_information
        self.timings['t_calculate_mutual_information'] = calculation_block.elapsed

        if self.parallel:
            sendcounts = self.comm.allgather(local_blocksize)
            displacements = self.comm.allgather(start)
            self.comm.Allgatherv(sendbuf=mutual_information_local,
                                 recvbuf=(mutual_information_global, sendcounts, displacements, MPI.DOUBLE))
        if self._file:
            HDF5_group = self._file[self._system_name]
            run_name = 'run_' + str(len(HDF5_group)) if run_name is None else run_name
            mi_dataset = HDF5_group.create_dataset(f'{run_name}/mutual_information', data=mutual_information_global)
            mi_dataset.attrs['rates'] = [rate[1] for rate in self._rates]
            mi_dataset.attrs['dt'] = self._dt

        return mutual_information_global

    def calculate_marginal_probability_evolution(self, molecules):
        """"""

        if self.P_trajectory is None:
            raise ValueError('No data found in self.results attribute.')
        point_maps = self._get_point_mappings(molecules)
        distribution: Dict[tuple, np.ndarray] = {}
        for point, map in point_maps.items():
            distribution[point] = np.array([ np.sum(self.P_trajectory[ts][map]) for ts in range(len(self.P_trajectory)) ])

        return distribution

    def calculate_average_population(self, species):
        """"""

        average_population = np.empty(shape=len(self.P_trajectory), dtype=np.float64)
        if self.P_trajectory is None:
            raise ValueError('No data found in self.results attribute.')
        P = self.P_trajectory
        point_maps = self._get_point_mappings(species)
        for ts in range(len(self.P_trajectory)):
            total = 0
            for point, map in point_maps.items():
                assert len(point) == 1
                count = point[0]
                count_term = np.sum( count*self.P_trajectory[ts][map] )
                total += count_term

            average_population[ts] = total

        return average_population

    def _get_Delta_vectors(self, molecules):

        if isinstance(molecules, str):
            molecules = [molecules]
        # indices that correspond to the selected molecular species in the ordered species list
        ids = [self.species.index(n) for n in molecules]
        truncated_points = np.array(self.states)[:, ids]

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
        truncated_points = np.array(self.states)[:, ids]

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

    def _calculate_matrix_element(self, i, j):
        """"""

        Pi = self.P_trajectory[i]
        Pj = self.P_trajectory[j]
        # time gap between i and j
        dt = abs(j - i) * self._dt
        # propagator for this dt
        Q = expm(self.G * dt)



    """def calculate_mutual_information_matrix(self, X, Y):
        Calculates the mutual information between every pair of timepoints

        n_timesteps = len(self.P)
        dts = [self._dt*timestep for timestep in range(n_timesteps)]
        dts_matrix = np.zeros(shape=(len(dts), len(dts)))
        for i, dt in enumerate(dts):
            np.fill_diagonal(dts_matrix[:-i, i:], dt)  # upper triangle
            np.fill_diagonal(dts_matrix[i:, :-i], dt)  # lower triangle

        MI_matrix = np.empty(shape=(len(dts), len(dts)), dtype=float)

        x_map = self._get_point_mappings(X)
        y_map = self._get_point_mappings(Y)
        xy_map = self._get_point_mappings(X + Y)

        Q_dict = {}
        for dt in dts:
            Q_dict[dt] = expm(self.G * dt)

        def calc_matrix_element(i, j):
            #print(f'\n\ndoing i = {i}, j = {j}')
            # p(n_A, n_B, n_C) at t=i
            P_i = self.P[i]
            # p(n_A, n_B, n_C) at t=j
            P_j = self.P[j]
            dt = dts_matrix[i, j]
            Q = Q_dict[dt]

            MI = 0
            for x_indices in x_map.values():
                for y_indices in y_map.values():
                    p_x = np.sum(P_i[x_indices])
                    p_y = np.sum(P_j[y_indices])
                    if i > j:
                        Qtilde = np.sum([Q[i, j] * P_j[j] for i in x_indices for j in y_indices])
                    elif i < j:
                        Qtilde = np.sum([Q[j, i] * P_i[i] for i in x_indices for j in y_indices])
                    if Qtilde == 0:
                        term = 0
                    else:
                        term = Qtilde * np.log(Qtilde / (p_x * p_y))
                    MI += term
                    #print(f'x point: {x_indices}\ny point: {y_indices}\nterm = {term}\n')
            return MI

        with timeit() as calculation_block:

            for i in ProgressBar(range(len(MI_matrix))):
                for j in range(len(MI_matrix)):
                    if i != j:
                        MI_matrix[i, j] = calc_matrix_element(i, j)

            main_diagonal = self.calculate_mutual_information(X, Y)
            np.fill_diagonal(MI_matrix, main_diagonal)

        return MI_matrix"""
