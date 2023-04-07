"""
Unit and regression test for the c3s package.
"""
import pytest
import sys
import numpy as np
import math
from c3s import ChemicalMasterEquation
from numpy.testing import assert_almost_equal, assert_equal, assert_array_almost_equal,assert_array_equal


def test_c3s_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "c3s" in sys.modules

class TestBinarySystem:

    @pytest.fixture(scope='class')
    def binary(self):
        system = ChemicalMasterEquation(cfg='config_files/binary.yml', initial_populations={'A':1})
        start, stop, step = 0, 1, 0.01
        system.run(start, stop, step)
        yield system

    @pytest.fixture(scope='class')
    def binary_many_body(self):
        system = ChemicalMasterEquation(cfg='config_files/binary.yml', initial_populations={'A':3})
        start, stop, step = 0, 1, 0.01
        system.run(start, stop, step)
        yield system

    def test_species(self, binary):
        correct_species = ['A', 'B']
        assert_equal(binary.species, correct_species)

    def test_rates(self, binary):
        correct_rates = {'k_1': 1, 'k_2': 1}
        assert_equal(binary.rates, correct_rates)

    def test_reaction_matrix(self,binary):
        correct_reaction_matrix = np.array([[-1, 1],
                                            [ 1,-1]])
        assert_equal(binary.reaction_matrix, correct_reaction_matrix)

    def test_states(self, binary):
        correct_states = np.array([[1,0],
                                   [0,1]])
        assert_equal(binary.states, correct_states)

    def test_G(self, binary):
        correct_G = np.array([[-1, 1],
                              [ 1,-1]])
        assert_equal(binary.G, correct_G)

    def test_changing_rates(self, binary):

        rate1, rate2 = 2, 2
        binary.update_rates(dict(k_1=rate1, k_2=rate2))
        assert_equal(binary.rates['k_1'], rate1)
        assert_equal(binary.rates['k_2'], rate2)
        assert_array_almost_equal(binary.G, np.array([[-2, 2],
                                                      [ 2,-2]]))
        rate3, rate4 = 6, 10
        binary.update_rates(dict(k_1=rate3, k_2=rate4))
        assert_equal(binary.rates['k_1'], rate3)
        assert_equal(binary.rates['k_2'], rate4)
        assert_array_almost_equal(binary.G, np.array([[-6, 10],
                                                      [ 6,-10]]))
        binary.reset_rates()
        assert_equal(binary.rates['k_1'], 1)
        assert_equal(binary.rates['k_2'], 1)
        assert_array_almost_equal(binary.G, np.array([[-1, 1],
                                                      [ 1,-1]]))

    def test_mutual_information(self, binary):

        X = 'A'
        Y = 'B'
        binary.calculate_instantaneous_mutual_information(X, Y, base=2)
        mut_inf = binary._mutual_information

        correct_values = np.array([0., 0.43858457, 0.6457636 , 0.77025155, 0.84901701,
                                   0.90004559, 0.93353366, 0.95567906, 0.9703932 , 0.98019935])
        assert_array_almost_equal(mut_inf[0::10], correct_values)

    def test_mutual_information_many_body(self, binary_many_body):

        X = 'A'
        Y = 'B'
        binary_many_body.calculate_instantaneous_mutual_information(X, Y, base=2)
        mut_inf = binary_many_body._mutual_information

        correct_values = np.array([0., 0.92385593, 1.28269609, 1.48006893, 1.59832796,
                                   1.67229091, 1.71971752, 1.75060155, 1.77091263, 1.78435647])

        assert_array_almost_equal(mut_inf[0::10], correct_values)


class Test2IsolatedSwitch:

    @pytest.fixture(scope='class')
    def isolated_switches(self):
        system = ChemicalMasterEquation(cfg='config_files/2_isolated_switches.yml', initial_populations={'A':1, 'B':1})
        start, stop, step = 0, 1, 0.01
        system.run(start, stop, step)
        yield system

    def test_mutual_information_many_body(self, isolated_switches):

        X = ['A', 'A*']
        Y = ['B', 'B*']
        mut_inf = isolated_switches.calculate_instantaneous_mutual_information(X, Y, base=2)
        correct_values = np.array([0.00000000e+00, 1.59908175e-15, 3.24752736e-15, 5.52514748e-15,
                                   7.38322748e-15, 9.14533561e-15, 1.12503565e-14, 1.35398846e-14,
                                   1.54150075e-14, 1.73809943e-14])
        assert_array_almost_equal(mut_inf[0::10], correct_values)


class TestAllosteryModel1:

    cfg = 'config_files/enzyme_no_allostery.yml'

    @pytest.fixture(scope='class')
    def allostery_model(self):
        system = ChemicalMasterEquation(cfg=self.cfg,
                                        initial_populations={'A':1, 'B':1, 'S': 1},
                                        max_populations={'S':1, 'P':1})
        start, stop, step = 0, 1, 0.01
        system.run(start, stop, step)
        yield system

    def test_species(self, allostery_model):
        correct_species = ['A', 'AS', 'B', 'BP', 'P', 'S']
        assert_equal(correct_species, allostery_model.species)

    def test_reaction_matrx(self, allostery_model):
        correct_matrx = np.array([
            [-1,  1,  0,  0,  0, -1],
            [ 1, -1,  0,  0,  0,  1],
            [ 1, -1,  0,  0,  1,  0],
            [ 0,  0, -1,  1, -1,  0],
            [ 0,  0,  1, -1,  1,  0],
            [ 0,  0,  0,  0, -1,  0],
            [ 0,  0,  0,  0,  0,  1]])
        assert_array_equal(correct_matrx, allostery_model.reaction_matrix)

    def test_max_population(self):

        maxS = [1, 2, 3]
        maxP = [1, 2, 3]

        for S, P in zip(maxS, maxP):
            system = ChemicalMasterEquation(cfg=self.cfg,
                                           initial_populations={'A':1, 'B':1, 'S': 1},
                                           max_populations={'S':S, 'P':P})
            i = system.species.index('S')
            j = system.species.index('P')
            assert np.max(system.states[:, i]) <= S
            assert np.max(system.states[:, j]) <= P
