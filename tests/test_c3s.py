import pytest
from c3s import ChemicalMasterEquation as CME
import math
from c3s.h5io import build_system_from_file
from numpy.testing import assert_equal, assert_array_almost_equal, assert_array_equal, assert_almost_equal
from .reference_data import RefBINARY, Ref2and2Iso, Ref4and2Iso, RefNoAllostery, RefAllosteric2State


class BaseTest:

    @pytest.fixture(scope='class')
    def correct_data(self):
        return self

    @pytest.fixture(scope='function')
    def system(self):
        return CME(config=self.config)

    @pytest.fixture(scope='function')
    def outfile(self, tmp_path):
        return tmp_path /'c3s_write_test.h5'

    def test_species(self, correct_data, system):
        assert_equal(system.species, correct_data.species)
        assert_equal(len(system.species), correct_data.N)

    def test_reactions(self, correct_data, system):
        for k, reaction in enumerate(system.reaction_network.reactions):
            assert_equal(reaction.k, k)
            assert_equal(reaction.reaction, correct_data.reactions[k])
            assert_equal(reaction.reactants, correct_data.reactants[k])
            assert_equal(reaction.products, correct_data.products[k])
            assert_equal(reaction.rate_name, correct_data.rate_names[k])
            assert_equal(reaction.rate, correct_data.rates[k])

    def test_reaction_matrix(self, correct_data, system):
        assert_array_equal(system.reaction_network.reaction_matrix, correct_data.reaction_matrix)
        assert_equal(system.K, correct_data.K)

    def test_species_in_reaction(self, correct_data, system):
        assert_equal(system.reaction_network.species_in_reaction, correct_data.species_in_reaction)

    def test_constraints(self, correct_data, system):
        for i, constraint in enumerate(system.reaction_network.constraints):
            assert_equal(constraint.species_involved, correct_data.constrained_species[i])
            assert_equal(constraint.separator, correct_data.constraint_separators[i])
            assert_equal(constraint.value, correct_data.constraint_values[i])

    def test_state_space(self, correct_data, system):
        assert_array_equal(system.states, correct_data.states)
        assert_equal(system.M, correct_data.M)

    def test_generator_matrix(self, correct_data, system):
        assert_array_equal(system.G.values, correct_data.G_sparse)
        assert_array_equal(system.G.to_dense(), correct_data.G_dense)

    def test_IMU_vs_EXPM(self):
        imu_system = CME(config=self.config)
        imu_system.run(method='IMU', N_timesteps=20)
        expm_sytem = CME(config=self.config)
        expm_sytem.run(method='EXPM', N_timesteps=20)
        imu_trajectory = imu_system.trajectory
        expm_trajectory = expm_sytem.trajectory
        assert_array_almost_equal(imu_trajectory, expm_trajectory, decimal=5)

    @pytest.mark.parametrize('method', ('EXPM', 'IMU'))
    def test_run_to_steady_state(self, correct_data, method):
        ss_system = CME(config=self.config)
        ss_system.run(method=method, to_steady_state=True)
        IMU_trajectory = ss_system.Trajectory.trajectory
        ss_system.run(method=method, to_steady_state=True, overwrite=True)
        EXPM_trajectory = ss_system.Trajectory.trajectory
        assert_array_almost_equal(IMU_trajectory, EXPM_trajectory)

    @pytest.mark.parametrize('method', ('EXPM', 'IMU'))
    def test_is_continued(self, correct_data, system, method):
        N_timesteps = 10
        system.run(method=method, N_timesteps=N_timesteps, overwrite=True)
        trajectory_contiguous = system.Trajectory.trajectory
        system.run(method=method, N_timesteps=5, overwrite=True)
        system.run(method=method, N_timesteps=5, is_continued_run=True)
        trajectory_continued = system.Trajectory.trajectory
        assert_array_equal(trajectory_contiguous, trajectory_continued)

    def test_file_io(self, system, outfile):
        timesteps = [5, 10, 15]
        dt = 1
        for N in timesteps:
            system.run(N_timesteps=N, dt=dt, overwrite=True, method='IMU')
            system.write.system_info(filename=outfile, mode='w')
            system.write.trajectory(filename=outfile, trajectory_name=f'{N}_timesteps')
            system2 = build_system_from_file(filename=outfile, trajectory_name=f'{N}_timesteps')
            assert_array_equal(system.states, system2.states)
            assert_array_equal(system.G.rows, system2.G.rows)
            assert_array_equal(system.G.columns, system2.G.columns)
            assert_array_equal(system.G.values, system2.G.values)
            assert_equal(system.species, system2.species)
            assert_array_equal(system.Trajectory.trajectory, system2.Trajectory.trajectory)

    def test_update_rates(self, correct_data, system):
        system.update_rates(self.updated_rates)
        for name, desired_rate in self.updated_rates.items():
            for reaction in system.reaction_network.reactions:
                if reaction.rate_name == name:
                    assert_equal(reaction.rate, desired_rate)
                    assert_equal(system._rates[reaction.k], desired_rate)

            assert_array_equal(system.G.values, self.G_sparse_updated)
            assert_array_equal(system.G.to_dense(), self.G_dense_updated)

    def test_initial_state(self, correct_data):
        system = CME(config=self.config, initial_populations=correct_data.initial_populations)
        system.run(N_timesteps=3, overwrite=True)
        assert_array_equal(system.states[system._initial_state_index], correct_data.initial_state)
        assert_equal(system._initial_state_index, correct_data.initial_state_index)
        assert_equal(system.trajectory[0][correct_data.initial_state_index], 1.0)

    def test_point_mappings(self, correct_data, system):
        system.run(N_timesteps=3, overwrite=True)
        point_mappings = system.calculate._get_point_mappings(self.species_subset)
        for key1, map1, key2 in zip(self.point_map_keys, self.point_map_ids, point_mappings.keys()):
            assert_array_equal(key1, key2)
            assert_array_equal(map1, point_mappings[key2])

    def test_marginalized_trajectory(self, correct_data, system):
        ...

    def test_average_copy_number(self, correct_data, system):
        ...

    def test_mutual_information(self, correct_data, system):
        system.run(N_timesteps=20, overwrite=True)
        for X, Y, truth_mi in zip(self.X_set, self.Y_set, self.ss_mutual_informations):
            mi = system.calculate.mutual_information(X, Y)
            assert_almost_equal(mi.data[-1], truth_mi)

class TestBinary(BaseTest, RefBINARY):
    pass

class Test2and2Iso(BaseTest, Ref2and2Iso):

    def test_analytic_mutual_information(self):
        def mi_term(p_xy, p_x, p_y):
            if p_xy == 0:
                return 0
            return p_xy * math.log2( p_xy / (p_x*p_y))

        system = CME(config=self.config)
        system.run(N_timesteps=20, dt=1)
        trajectory = system.trajectory
        exp_MI = system.calculate.mutual_information(X=['A', 'B'], Y=['X', 'Y'])
        mi = []
        for P in trajectory:
            mi_sum = 0
            p_1, p_2, p_3, p_4 = P
            mi_sum += mi_term(p_4, p_4+p_3, p_4+p_2) + mi_term(p_2, p_2+p_1, p_4+p_2) + mi_term(p_3, p_4+p_3, p_3+p_1) + mi_term(p_1, p_2+p_1, p_3+p_1)
            mi.append(mi_sum)

        assert_array_almost_equal(mi, exp_MI.data)

class Test4and2Iso(BaseTest, Ref4and2Iso):
    def test_update_rates(self):
        ...
    def test_point_mappings(self):
        ...
    pass

class TestNoAllostery(BaseTest, RefNoAllostery):
    def test_update_rates(self):
        ...
    def test_point_mappings(self):
        ...
    def test_mutual_information(self):
        ...

class TestAllosteric2State(BaseTest, RefAllosteric2State):
    def test_state_space(self):
        ...
    def test_generator_matrix(self):
        ...
    def test_update_rates(self):
        ...
    def test_point_mappings(self):
        ...
    def test_mutual_information(self):
        ...