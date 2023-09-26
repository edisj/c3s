"""
Unit and regression test for the c3s package.
"""
import pytest
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
            assert_equal(reaction.reactants, self.reactants[k])
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
    def test_is_continued(self, system, method):
        N_timesteps = 10
        dt = 1
        system.run(dt=dt, method=method, N_timesteps=N_timesteps, overwrite=True)
        trajectory_contiguous = system.Trajectory.trajectory
        system.run(dt=dt, method=method, N_timesteps=5, overwrite=True)
        system.run(dt=dt, method=method, N_timesteps=5, is_continued_run=True)
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

    def test_update_rates(self, system):
        system.update_rates(self.updated_rates)
        for name, desired_rate in self.updated_rates.items():
            for reaction in system.reaction_network.reactions:
                if reaction.rate_name == name:
                    assert_equal(reaction.rate, desired_rate)
                    assert_equal(system._rates[reaction.k], desired_rate)

            assert_array_equal(system.G.values, self.G_sparse_updated)
            assert_array_equal(system.G.to_dense(), self.G_dense_updated)

    def test_initial_state(self, system):
        ...

    def test_marginalized_trajectory(self, system):
        ...

    def test_mutual_information(self, system):
        ...


class TestBinary(BaseTest, RefBinary):
    pass


class Test2and2Iso(BaseTest, Ref2and2Iso):
    pass


class Test4and2Iso(BaseTest, Ref4and2Iso):
    def test_update_rates(self):
        pass

class TestMMEnzyme:
    ...

class TestA2SM:
    ...
