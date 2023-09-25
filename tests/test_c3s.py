"""
Unit and regression test for the c3s package.
"""
import pytest
#import numpy as np
#import math
from c3s import ChemicalMasterEquation as CME
from c3s.h5io import build_system_from_file
from numpy.testing import assert_equal, assert_array_almost_equal,assert_array_equal#, assert_almost_equal
from .reference_data import RefBinary, Ref2and2Iso, Ref4and2Iso


class BaseTest:

    @pytest.fixture()
    def system(self):
        return CME(config=self.reactions_file)

    @pytest.fixture()
    def outfile(self, tmpdir):
        yield str(tmpdir) + 'c3s_write_test.h5'

    def test_species(self, system):
        assert_equal(system.species, self.species)
        assert_equal(len(system.species), self.N)

    def test_reactions(self, system):
        for k, reaction in enumerate(system.reaction_network.reactions):
            assert_equal(reaction.k, k)
            assert_equal(reaction.reaction, self.reactions[k])
            assert_equal(reaction.reactants, self.reactants[k] )
            assert_equal(reaction.products, self.products[k])
            assert_equal(reaction.rate_name, self.rate_names[k])
            assert_equal(reaction.rate, self.rates[k])

    def test_reaction_matrix(self, system):
        assert_array_equal(system.reaction_network.reaction_matrix, self.reaction_matrix)
        assert_equal(system.K, self.K)

    def test_species_in_reaction(self, system):
        assert_equal(system.reaction_network.species_in_reaction, self.species_in_reaction)

    def test_constraints(self, system):
        for i, constraint in enumerate(system.reaction_network.constraints):
            assert_equal(constraint.species_involved, self.constrained_species[i])
            assert_equal(constraint.separator,self.constraint_separators[i])
            assert_equal(constraint.value, self.constraint_values[i])

    def test_state_space(self, system):
        assert_array_equal(system.states, self.states)
        assert_equal(system.M, self.M)

    def test_generator_matrix(self, system):
        assert_array_equal(system.G.values, self.G_sparse)
        assert_array_equal(system.G.to_dense(), self.G_dense)

    def test_IMU_vs_EXPM(self, system):
        N_timesteps = 15
        dt = 1
        system.run(dt=dt, method='IMU', N_timesteps=N_timesteps)
        IMU_trajectory = system.Trajectory.trajectory
        system.run(dt=dt, method='EXPM', N_timesteps=N_timesteps, overwrite=True)
        EXPM_trajectory = system.Trajectory.trajectory
        assert_array_almost_equal(IMU_trajectory, EXPM_trajectory, decimal=5)

    @pytest.mark.parametrize('method', ('EXPM', 'IMU'))
    def test_continued(self, system, method):
        N_timesteps = 10
        dt = 1
        system.run(dt=dt, method=method, N_timesteps=N_timesteps, overwrite=True)
        trajectory_contiguous = system.Trajectory.trajectory
        system.run(dt=dt, method=method, N_timesteps=5, overwrite=True)
        system.run(dt=dt, method=method, N_timesteps=5, continued=True)
        trajectory_continued = system.Trajectory.trajectory
        assert_array_equal(trajectory_contiguous, trajectory_continued)

    def test_file_io(self, system, outfile):

        timesteps = [5, 10, 15]
        dt = 1
        for N in timesteps:
            system.run(N_timesteps=N, dt=dt, overwrite=True, method='IMU')
            system.write_system(filename=outfile, mode='w')
            system.write_trajectory(filename=outfile, trajectory_name=f'{N}_timesteps')
            system2 = build_system_from_file(filename=outfile, trajectory_name=f'{N}_timesteps')
            assert_array_equal(system.states, system2.states)
            assert_array_equal(system.G.rows, system2.G.rows)
            assert_array_equal(system.G.columns, system2.G.columns)
            assert_array_equal(system.G.values, system2.G.values)
            assert_equal(system.species, system2.species)
            assert_array_equal(system.Trajectory.trajectory, system2.Trajectory.trajectory)

    def test_reset_rates(self, system):
        ...

    def test_initial_state(self, system):
        ...


class TestBinary(BaseTest, RefBinary):
    def test_update_rates(self, system):
        system.update_rates(self.updated_rates)
        for reaction, desired_rate in zip(system.reaction_network.reactions, self.updated_rates.values()):
            assert_equal(reaction.rate, desired_rate)
            assert_array_equal(system.G.values, self.G_sparse_updated)
            assert_array_equal(system.G.to_dense(), self.G_dense_updated)
    '''def test_mutual_information(self):
        X = 'A'
        Y = 'B'
        correct_values = np.array([0., 0.43858457, 0.6457636 , 0.77025155, 0.84901701,
                                   0.90004559, 0.93353366, 0.95567906, 0.9703932 , 0.98019935])

    def test_mutual_information_many_body(self):
        X = 'A'
        Y = 'B'
        correct_values = np.array([0., 0.92385593, 1.28269609, 1.48006893, 1.59832796,
                                   1.67229091, 1.71971752, 1.75060155, 1.77091263, 1.78435647])'''


class Test2and2Iso(BaseTest, Ref2and2Iso):
    pass
    '''def test_mutual_information_iso_switches(self, isolated_switches):

        X = ['A', 'A*']
        Y = ['B', 'B*']
        isolated_switches.calculate_instantaneous_mutual_information(X, Y, base=2)
        mut_inf = isolated_switches._mutual_information
        correct_values = np.array([0.00000000e+00, 1.59908175e-15, 3.24752736e-15, 5.52514748e-15,
                                   7.38322748e-15, 9.14533561e-15, 1.12503565e-14, 1.35398846e-14,
                                   1.54150075e-14, 1.73809943e-14])
        assert_array_almost_equal(mut_inf[0::10], correct_values)'''


class Test4and2Iso(BaseTest, Ref4and2Iso):
    pass
