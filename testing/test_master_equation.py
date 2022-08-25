"""
Unit and regression test for the c3s package.
"""
import sys
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_array_almost_equal
import numpy as np
import c3s
from c3s.calculations import _calc_mutual_information_OLD_VERSION


def test_c3s_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "c3s" in sys.modules


class TestMutualInformationWithSyntheticData:

    @pytest.fixture(scope='class')
    def synthetic_system(self):
        empty_system = c3s.simulators.simulators.MasterEquation(empty=True)

        # make up fake species
        species = ['X', 'Y', 'Z']
        # label constitutive states
        constitutive_states = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1),
                               (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
        # create synthetic results
        P0 = np.array([1]*8) / 8
        P1 = np.array([10000, 1, 1, 1, 1, 1, 1, 10000]) / sum([10000, 1, 1, 1, 1, 1, 1, 10000])
        results = np.empty(shape=(2, 8))
        results[0] = P0
        results[1] = P1
        # fill in the system with synthetic data
        empty_system.species = species
        empty_system.constitutive_states = constitutive_states
        empty_system.results = results

        yield empty_system

    def test_mutual_information(self, synthetic_system):

        MI_0 = 0              # 1st timestep
        MI_1 = 0.691244272798 # 2nd timestep

        MIs = synthetic_system.calculate_mutual_information(['X'], ['Y'])

        assert_almost_equal(MI_0, MIs[0], decimal=5)
        assert_almost_equal(MI_1, MIs[1], decimal=5)


class TestABCToyModel:

    @pytest.fixture(scope='class')
    def ABC_system(self):
        system = c3s.simulators.simulators.MasterEquation(initial_species={'A': 2}, cfg='ABC.cfg')
        start, stop, step = 0, 0.1, 0.01
        system.run(start=start, stop=stop, step=step)
        yield system

    def test_mutual_information(self, ABC_system):

        correct_MIs = np.array([0., 0.11230703, 0.18298896, 0.23830284, 0.28390703,
                                0.32250281, 0.35568427, 0.38450472, 0.40971173, 0.43186306])

        MIs = ABC_system.calculate_mutual_information(['A'], ['C'])

        assert_array_almost_equal(MIs, correct_MIs, decimal=5)

    def test_mutual_information_matrix(self, ABC_system):

        correct_matrix = np.array([
            [0.00000000e+00, 0.00000000e+00, -2.15890371e-16,
             -1.84902389e-16, -2.72613542e-16, -5.18488199e-16,
             -5.01693776e-16, -4.04592802e-16, -5.72945086e-16,
             -6.51276856e-16],
            [0.00000000e+00, 1.12307026e-01, 7.11521814e-02,
             5.59958564e-02, 4.64679775e-02, 3.96428010e-02,
             3.44206972e-02, 3.02611075e-02, 2.68560740e-02,
             2.40126714e-02],
            [-2.22372659e-16, 7.89371126e-02, 1.82988962e-01,
             1.27041241e-01, 1.02451291e-01, 8.61401366e-02,
             7.41123046e-02, 6.47384738e-02, 5.71752553e-02,
             5.09245374e-02],
            [-1.97545646e-16, 6.29313066e-02, 1.39557540e-01,
             2.38302837e-01, 1.73311849e-01, 1.42045506e-01,
             1.20582703e-01, 1.04429804e-01, 9.16661691e-02,
             8.12637144e-02],
            [-3.80500145e-16, 5.24130209e-02, 1.14292298e-01,
             1.88979886e-01, 2.83907034e-01, 2.12652752e-01,
             1.76352540e-01, 1.50823597e-01, 1.31313859e-01,
             1.15730040e-01],
            [-4.19654858e-16, 4.47240152e-02, 9.65851221e-02,
             1.57491545e-01, 2.30505494e-01, 3.22502806e-01,
             2.46646701e-01, 2.06407442e-01, 1.77587418e-01,
             1.55295933e-01],
            [-4.64619075e-16, 3.87805232e-02, 8.31980960e-02,
             1.34505578e-01, 1.94499643e-01, 2.66035984e-01,
             3.55684272e-01, 2.76336678e-01, 2.32941293e-01,
             2.01411119e-01],
            [-4.34150971e-16, 3.40235914e-02, 7.26313688e-02,
             1.16712318e-01, 1.67464109e-01, 2.26593700e-01,
             2.96802252e-01, 3.84504717e-01, 3.02460967e-01,
             2.56496930e-01],
            [-6.54648769e-16, 3.01236811e-02, 6.40515484e-02,
             1.02446046e-01, 1.46165066e-01, 1.96359681e-01,
             2.54664083e-01, 3.23662694e-01, 4.09711732e-01,
             3.25568925e-01],
            [-5.83299280e-16, 2.68688856e-02, 5.69424696e-02,
             9.07311306e-02, 1.28878238e-01, 1.72214575e-01,
             2.21851656e-01, 2.79364240e-01, 3.47248043e-01,
             4.31863062e-01]])

        MI_matrix = ABC_system.calculate_mutual_information_matrix(['A'], ['C'])

        assert_array_almost_equal(MI_matrix, correct_matrix, decimal=5)


class TestMasterEquationToyModel:

    @pytest.fixture(scope='class')
    def test_system(self):
        system = c3s.simulators.simulators.MasterEquation(initial_species={'A': 2}, cfg='config_for_tests.cfg')
        start, stop, step = 0, 3, 0.001
        system.run(start=start, stop=stop, step=step)
        yield system

    def test_species(self, test_system):

        species = ['A', 'A*', 'B']
        assert_equal(species, test_system.species)

    def test_reaction_matrix(self, test_system):

        correct_reaction_matrix = np.array(
            [
                [-1,  0,  1],
                [ 1,  0, -1],
                [-1,  1,  0],
                [ 1, -1,  0],
                [ 0, -1,  1],
                [ 0,  1, -1]
            ]
        )

        assert_equal(correct_reaction_matrix, test_system.reaction_matrix)

    def test_generator_matrix(self, test_system):

        correct_generator_matrix = [
            [-8, 2, 1, 0, 0, 0],
            [6, -10, 6, 4, 1, 0],
            [2, 4, -11, 0, 2, 2],
            [0, 3, 0, -12, 6, 0],
            [0, 1, 3, 8, -13, 12],
            [0, 0, 1, 0, 4, -14]]

        test_m = test_system.generator_matrix
        assert_equal(correct_generator_matrix, test_m)

    def test_update_rates(self, test_system):

        rate_from_config = 3
        new_rate = 10

        assert_equal(test_system.rates[0], rate_from_config)
        test_system.update_rates({test_system.rate_strings[0]: new_rate})
        assert_equal(test_system.rates[0], new_rate)

    def test_reset_rates(self, test_system):

        rate_from_config = 3
        new_rate = 10
        test_system.update_rates({test_system.rate_strings[0]: new_rate})

        correct_generator_matrix = [
            [-8, 2, 1, 0, 0, 0],
            [6, -10, 6, 4, 1, 0],
            [2, 4, -11, 0, 2, 2],
            [0, 3, 0, -12, 6, 0],
            [0, 1, 3, 8, -13, 12],
            [0, 0, 1, 0, 4, -14]]

        test_system.reset_rates()
        test_matrix = test_system.generator_matrix
        assert_equal(test_system.rates[0], rate_from_config)
        assert_equal(correct_generator_matrix, test_matrix)

    def test_mutual_information_old(self, test_system):

        MI_1 = 0.02039576       # 1st timestep
        MI_500 = 1.04275917     # 500th timestep
        MI_1000 = 1.08477599    # 1000th timestep
        MI_last = 1.09005295    # last timestep

        A, B = ['A', 'A*'], ['B']
        MIs = _calc_mutual_information_OLD_VERSION(test_system, A, B)

        assert_almost_equal(MI_1, MIs[1])
        assert_almost_equal(MI_500, MIs[500])
        assert_almost_equal(MI_1000, MIs[1000])
        assert_almost_equal(MI_last, MIs[-1])

    def test_mutual_information_new(self, test_system):

        MI_1 = 0.02039576  # 1st timestep
        MI_500 = 1.04275917  # 500th timestep
        MI_1000 = 1.08477599  # 1000th timestep
        MI_last = 1.09005295  # last timestep

        A, B = ['A', 'A*'], ['B']
        #matrix = test_system.generator_matrix
        #print(f'{test_system.size}, {test_system.parallel}')
        if test_system.rank == 0:
            print(f'rank {test_system.rank} has\n',
                  f'{test_system.constitutive_states}\n',
                  f'{test_system.generator_matrix}')

        MIs = test_system.calculate_mutual_information(A, B)

        assert_almost_equal(MI_1, MIs[1])
        assert_almost_equal(MI_500, MIs[500])
        assert_almost_equal(MI_1000, MIs[1000])
        assert_almost_equal(MI_last, MIs[-1])
