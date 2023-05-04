from collections import namedtuple
import math
import random
import numpy as np
from .reactions import Reactions



class Gillespie:
    """The Gillespie stochastic simulation algorithm (SSA)."""

    def __init__(self, config, initial_state=None, initial_populations=None, max_populations=None, empty=False):
        """Uses Gillespie's stochastic simulation algorithm to generate a trajectory of a random walker that is
        defined by a molecular population vector.

        Args:
            config:
                Path to yaml config file that specifies chemical reactions and kinetic rates
            initial_state:
            initial_populations:
                The initial population a particular species. If a species population is
                not specified, it's initial population is taken to be 0.
            max_populations:
            empty:

        """

        self.reactions = Reactions(config)
        self.species = self.reactions.species
        self.rates = self.reactions.rates
        self._rates = self.reactions._rates
        self._initial_state = initial_state
        self._initial_populations = initial_populations
        self._max_populations = max_populations
        self.empty = empty

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

    def _get_propensity_vector(self, currState):
        """"""
        reaction_propensities = [np.prod(currState[indices])*rate[1]
                                 for indices, rate in zip(self.reactions.propensity_ids, self.reactions._rates)]

        return np.array(reaction_propensities, dtype=np.int32)

    def _get_next_state(self, currState, propensity_vector):
        """"""

        # k selects the index of which reaction was sampled to fire
        k = self._sample_categorical(propensity_vector)
        reaction = self.reactions.reaction_matrix[k]
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

        if self.trajectory is not None and not overwrite:
            raise ValueError("Data from previous run found in `self.trajectory`. "
                             "To write over this data, set the `overwrite=True`")

        N_species = len(self.species)
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

        self._trajectory = trajectory

    def run_many_iterations(self, N_iterations, N_timesteps, run_name=None):
        """

        Parameters
        ----------
        N_iterations : int
        N_timesteps  : int

        """

        N_species = len(self.species)
        trajectories = np.empty(shape=(N_iterations, N_timesteps, N_species), dtype=np.int32)
        times = np.empty(shape=(N_iterations, N_timesteps), dtype=np.float64)

        for i in range(N_iterations):
            self.run(N_timesteps, run_name=run_name, overwrite=True)
            trajectories[i] = self.trajectory.states
            times[i] = self.trajectory.times

        self._trajectory = (trajectories, times)

    @property
    def trajectory(self):
        return self._trajectory
    @trajectory.setter
    def trajectory(self, value):
        self._trajectory = value
    @property
    def Trajectories(self):
        return self._trajectory
    @Trajectories.setter
    def Trajectories(self, value):
        self._trajectory = value