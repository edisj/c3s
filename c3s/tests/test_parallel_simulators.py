import pytest
from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from c3s.parallel_simulators import MasterEquationParallel
from c3s.calculations import calc_mutual_information_old

class TestParallelSimulators:

    @pytest.fixture(scope='class')
    def test_system(self):
        system = MasterEquationParallel(initial_species={'A': 2}, cfg='test_reactions.cfg')
        start, stop, step = 0, 3, 0.001
        system.run(start, stop, step)
        yield system

    def test_species(self, test_system):

        species = ['A', 'A*', 'B']
        assert_equal(species, test_system.species)

    def test_reaction_matrix(self, test_system):

        reaction_matrix = np.array(
            [[-1,  0,  1],
            [ 1,  0, -1],
            [-1,  1,  0],
            [ 1, -1,  0],
            [ 0, -1,  1],
            [ 0,  1, -1]])

        assert_equal(reaction_matrix, test_system.reaction_matrix)

    def test_generator_matrix(self, test_system):

        generator_matrix = [
            [-4, 3, 1, 0, 0, 0],
            [2, -10, 4, 3, 1, 0],
            [1, 6, -11, 0, 3, 1],
            [0, 2, 0, -6, 4, 0],
            [0, 1, 2, 6, -13, 4],
            [0, 0, 1, 0, 6, -7]]

        print(test_system.generator_matrix)
        assert_equal(generator_matrix, test_system.generator_matrix)

    def test_changing_rates(self, test_system):

        rate_from_config = 3
        new_rate = 10

        assert_equal(test_system.rates[0], rate_from_config)
        test_system.update_rates({test_system.rate_strings[0]: new_rate})
        assert_equal(test_system.rates[0], new_rate)
        test_system.reset_rates()
        assert_equal(test_system.rates[0], rate_from_config)

    def test_mutual_information(self, test_system):

        MI_1 = 0.02039576       # 1st timestep
        MI_500 = 1.04275917     # 500th timestep
        MI_1000 = 1.08477599    # 1000th timestep
        MI_last = 1.09005295    # last timestep

        A, B = ['A', 'A*'], ['B']
        MIs = calc_mutual_information_old(test_system, A, B)

        assert_almost_equal(MI_1, MIs[1])
        assert_almost_equal(MI_500, MIs[500])
        assert_almost_equal(MI_1000, MIs[1000])
        assert_almost_equal(MI_last, MIs[-1])
