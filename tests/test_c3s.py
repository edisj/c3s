"""
Unit and regression test for the c3s package.
"""
import pytest
import sys
import numpy as np
import math
from c3s import ChemicalMasterEquation as CME
from c3s.h5io import read, write
from numpy.testing import assert_almost_equal, assert_equal, assert_array_almost_equal,assert_array_equal
from .reference import RefBinary


def test_c3s_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "c3s" in sys.modules


class BaseTest:

    @pytest.fixture()
    def system(self):
        return CME(config=self.reactions_file)

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
        ...

    def test_changing_rates(self, system):
        ...

    def test_reset_rates(self, system):
        ...


class TestBinary(BaseTest, RefBinary):

    def test_mutual_information(self):
        X = 'A'
        Y = 'B'
        correct_values = np.array([0., 0.43858457, 0.6457636 , 0.77025155, 0.84901701,
                                   0.90004559, 0.93353366, 0.95567906, 0.9703932 , 0.98019935])

    def test_mutual_information_many_body(self):
        X = 'A'
        Y = 'B'
        correct_values = np.array([0., 0.92385593, 1.28269609, 1.48006893, 1.59832796,
                                   1.67229091, 1.71971752, 1.75060155, 1.77091263, 1.78435647])

    def test_continued(self):
        ...


'''class TestH5IO:

    @pytest.fixture()
    def outfile(self, tmpdir):
        yield str(tmpdir) + 'c3s_write_test.h5'

    def test_file_io(self, outfile):
        system = ChemicalMasterEquation(config='config_files/2_isolated_switches.yml',
                                        initial_populations={'A': 1, 'B': 1})
        system.run()

        system.write(outfile)
        system2 = read(outfile)
        assert_array_equal(system.states, system2.states)
        assert_array_equal(system.G, system2.G)
        assert_equal(system.species, system2.species)

        stops = [1, 2, 3]
        for stop in stops:
            system.run(0, stop, 0.01, overwrite=True)
            system.write(outfile, trajectory_name=f'{stop}')
            system3 = read(outfile, trajectory_name=f'{stop}')
            assert_array_equal(system.trajectory, system3.trajectory)'''


'''class Test2IsolatedSwitch:

    def test_mutual_information_iso_switches(self, isolated_switches):

        X = ['A', 'A*']
        Y = ['B', 'B*']
        isolated_switches.calculate_instantaneous_mutual_information(X, Y, base=2)
        mut_inf = isolated_switches._mutual_information
        correct_values = np.array([0.00000000e+00, 1.59908175e-15, 3.24752736e-15, 5.52514748e-15,
                                   7.38322748e-15, 9.14533561e-15, 1.12503565e-14, 1.35398846e-14,
                                   1.54150075e-14, 1.73809943e-14])
        assert_array_almost_equal(mut_inf[0::10], correct_values)'''