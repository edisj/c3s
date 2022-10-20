import math
import yaml
import copy
import h5py
import random
import numpy as np
from typing import List, Dict
from collections import namedtuple
from scipy.sparse.linalg import expm
from mpi4py import MPI
from ..simulators import Gillespie, ChemicalMasterEquation
from ..utils import timeit, ProgressBar, split_tasks_for_workers


class GillespieParallel(Gillespie):
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

        super(GillespieParallel, self).__init__(cfg=cfg, system_name=system_name,
                                        initial_state=initial_state, initial_populations=initial_populations,
                                        max_populations=max_populations, empty=empty)

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

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
        start, stop, blocksize = split_tasks_for_workers(N_tasks=N_iterations, N_workers=self.size, rank=self.rank)
        trajectories = np.empty(shape=(blocksize, N_timesteps, N_species), dtype=np.int32)
        times = np.empty(shape=(blocksize, N_timesteps), dtype=np.float64)

        for iter in range(start, stop):
            self.run(N_timesteps, run_name=run_name, overwrite=True)
            trajectories[iter] = self.trajectory.states
            times[iter] = self.trajectory.times

        trajectories_global = np.empty(shape=(N_iterations, N_timesteps, N_species), dtype=np.int32)
        traj_sendcounts = self.comm.allgather(trajectories.size)
        traj_displacements = self.comm.allgather(self.rank * trajectories.size)
        self.comm.Allgatherv(
            sendbuf=trajectories, recvbuf=(trajectories_global, traj_sendcounts, traj_displacements, MPI.INT))
        holding_times_global = np.empty(shape=(N_iterations, N_timesteps - 1), dtype=np.float64)
        h_sendcounts = self.comm.allgather(times.size)
        h_displacements = self.comm.allgather(self.rank * times.size)
        self.comm.Allgatherv(
            sendbuf=times, recvbuf=(holding_times_global, h_sendcounts, h_displacements, MPI.DOUBLE))
        self._results = (trajectories_global, holding_times_global)

    def _write_to_file(self):
        this_run_group = self._system_file.create_group(self._run_name)
        trajectory_dset = this_run_group.create_dataset('trajectories', shape=self._results[0].shape, dtype=self._results[0].dtype)
        trajectory_dset.attrs['rates'] = [rate[1] for rate in self._rates]
        holding_times_dset = this_run_group.create_dataset('times', shape=self._results[1].shape, dtype=self._results[1].dtype)

        with trajectory_dset.collective:
            trajectory_dset[:] = self._results[0]
        with holding_times_dset.collective:
            holding_times_dset[:] = self._results[1]


class ChemicalMasterEquationParallel(ChemicalMasterEquation):
    """Simulator class of the Chemical Master Equationn (CME)."""

    def __init__(self, cfg=None, filename=None, mode='x', system_name=None,
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
        super(ChemicalMasterEquationParallel, self).__init__(cfg=cfg, filename=filename, mode=mode, system_name=system_name,
                                                             initial_state=initial_state, initial_populations=initial_populations,
                                                             max_populations=max_populations, empty=empty)
        self._constitutive_states = None
        self._generator_matrix = None
        self._max_populations = max_populations

        if not empty:
            with timeit() as set_constitutive_states:
                self._set_constitutive_states()
            with timeit() as set_generator_matrix:
                self._set_generator_matrix()
            self.timings['t_set_constitutive_states'] = set_constitutive_states.elapsed
            self.timings['t_set_generator_matrix'] = set_generator_matrix.elapsed

    def _set_generator_matrix(self):
        """Constructs the generator matrix.

        The generator matrix is an MxM matrix where M is the total number of states given
        by `len(self.constitutive_states)`.

        """

        M = len(self._constitutive_states)
        K = len(self._reaction_matrix)
        start, stop, blocksize = split_tasks_for_workers(N_tasks=M, N_workers=self.size, rank=self.rank)

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
                            self._G_propensity_ids[k].append((global_i, j))
                            break

    def run(self, start, stop, step, run_name=None, continued=False, overwrite=False):
        """Runs the chemical master equation simulation.

        Parameters
        ----------
        start : int or float
        stop  : int or float
        step  : int or float
        run_name : str
        overwrite : bool

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
        if self.rank == 0:
            with timeit() as matrix_exponential:
                Q = expm(self._generator_matrix * self._dt)
            with timeit() as run_time:
                for ts in ProgressBar(range(n_timesteps - 1), desc=f'rank {self.rank} running.'):
                    P_trajectory[ts + 1] = Q.dot(P_trajectory[ts])
            self.timings['t_matrix_exponential'] = matrix_exponential.elapsed
            self.timings['t_run'] = run_time.elapsed

        self._results = P_trajectory

    def _write_to_file(self, states=False, G=False, results=False, mutinf=False):
        this_run_group = self._system_file.create_group(self._run_name)
        #P_dset = this_run_group.create_dataset('P_trajectory', shape=self._results.shape, dtype=self._results.dtype)
        #P_dset.attrs['rates'] = [rate[1] for rate in self._rates]
        #P_dset.attrs['dt'] = self._dt
        #G_dset = this_run_group.create_dataset('G', shape=self._generator_matrix.shape, dtype=self._generator_matrix.dtype)
        #states_dset = this_run_group.create_dataset('constitutive_states', shape=self._constitutive_states.shape, dtype=self._constitutive_states.dtype)

        #if self.parallel:
            #with P_dset.collective:
            #    P_dset[:] = self._results
            #with G_dset.collective:
            #    G_dset[:] = self._generator_matrix
            #with states_dset.collective:
            #    states_dset[:] = self._constitutive_states
        #else:
        #    P_dset[:] = self._results
        #    G_dset[:] = self._generator_matrix
        #    states_dset[:] = self._constitutive_states
