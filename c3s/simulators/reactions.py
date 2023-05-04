import yaml
import copy
from typing import List, Dict, Union
from pathlib import Path
import numpy as np


class Reactions:
    """meant to be used as a component class via composition for the simulator classes"""

    def __init__(self, config):
        """reads the config file and sets various important attributes"""

        # for reading configs from h5 file
        if isinstance(config, dict):
            self._original_config = config
        else:
            config = Path(config)
            with open(config) as yaml_file:
                self._original_config = yaml.load(yaml_file, Loader=yaml.Loader)
        self._rates, self._reaction_strings, self._reactants, self._products = self._set_rates()
        self._species = self._set_species_vector()
        self._reaction_matrix = self._set_reaction_matrix()
        self._propensity_ids = self._set_reaction_propensities()

    @property
    def rates(self):
        """`self._rates` is len(K) List[List[str, float]] where k'th element gives
        the name and value of the rate constant for the k'th reaction
        """
        return dict(self._rates)
    @rates.setter
    def rates(self, value):
        self._rates = value

    @property
    def species(self) -> List[str]:
        """`self.species` is len(N) List[str] where n'th element is the name of the n'th species"""
        return self._species
    @species.setter
    def species(self, value):
        self._species = value

    @property
    def reaction_matrix(self) -> np.ndarray:
        """`self.reaction_matrix` is shape(K,N) array where the [k,n] element
        gives the change in the n'th species for the k'th reaction"""
        return self._reaction_matrix
    @reaction_matrix.setter
    def reaction_matrix(self, value):
        self._reaction_matrix = value

    @property
    def propensity_ids(self):
        """`self._propensenity_ids` is len(K) List[List[int]] whose k'th element
         gives the indices of `self.species` that are involved in the k'th reaction"""
        return self._propensity_ids
    @propensity_ids.setter
    def propensity_ids(self, value):
        self._propensity_ids = value

    def _set_rates(self):
        rates, reaction_strings, reactants, products = [], [], [], []
        # deepcopy because update_rates() would otherwise change rates in self._original_config
        config_data = copy.deepcopy(self._original_config)
        for reaction, rate_list in config_data['reactions'].items():
            rates.append(rate_list)
            reaction_strings.append(reaction)
            reactants.append(reaction.replace(' ', '').split('->')[0].split('+'))
            products.append(reaction.replace(' ', '').split('->')[1].split('+'))
        return rates, reaction_strings, reactants, products

    def _set_species_vector(self):
        species = []
        # TODO: need to add check and warning for birth process without max_popoulations
        for k, (reactants, products) in enumerate(zip(self._reactants, self._products)):
            # len(reactants) is not necessarily = len(products) so we have to loop over each
            for molecule in reactants:
                species.append(molecule)
            for molecule in products:
                species.append(molecule)
        while '0' in species:
            species.remove('0')
        # remove duplicates and sort
        species = sorted(list(set(species)))
        return species

    def _set_reaction_matrix(self):
        N_reactions = len(self._reactants)
        N_species = len(self._species)
        reaction_matrix = np.zeros(shape=(N_reactions, N_species), dtype=np.int32)
        for reaction, reactants, products in zip(reaction_matrix, self._reactants, self._products):
            for n, species in enumerate(self._species):
                # if this species in both a product and reactant, net effect is 0
                if species in reactants:
                    reaction[n] += -1
                if species in products:
                    reaction[n] += 1
        return reaction_matrix

    def _set_reaction_propensities(self):
        reaction_matrix = self._reaction_matrix
        N, K = len(self._species), len(reaction_matrix)
        propensity_ids = [[n for n in range(N) if reaction_matrix[k, n] < 0] for k in range(K)]
        return propensity_ids

    def print_propensities(self):
        """generates a readable list of the propensity of each reaction"""

        propensity_strings: List[str] = []
        for propensity_ids, rate in zip(self._propensity_ids, self._rates):
            transition_rate = rate[0]
            for n in propensity_ids:
                transition_rate += f'c^{self._species[n]}'
            propensity_strings.append(transition_rate)

        return propensity_strings
