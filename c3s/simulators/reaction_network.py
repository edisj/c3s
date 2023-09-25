from collections import namedtuple
from pathlib import Path
from typing import List
from copy import deepcopy
import yaml
import numpy as np


class ReactionNetwork:
    """meant to be used as a component class via composition for the simulator classes"""

    Reaction = namedtuple("Reaction", ['k', 'reaction', 'reactants', 'products', 'rate_name', 'rate'])
    Constraint = namedtuple("Constraint", ['constraint', 'species_involved', 'separator', 'value'])

    def __init__(self, config):
        """reads the config file and sets various important attributes"""

        if isinstance(config, dict):
            # for reading configs from h5 file
            self._original_config = config
        else:
            config = Path(config)
            with open(config) as yaml_file:
                self._original_config = yaml.load(yaml_file, Loader=yaml.Loader)
        self._reactions = self._parse_reactions()
        self._constraints = self._parse_constraints()
        self._rates = [reaction.rate for reaction in self._reactions]
        self._rate_names = [reaction.rate_name for reaction in self._reactions]
        self._species = self._set_species_vector()
        self._reaction_matrix = self._set_reaction_matrix()
        self._species_in_reaction = self._set_species_in_reaction()

    @property
    def reactions(self):
        return self._reactions
    @property
    def constraints(self):
        return self._constraints
    @property
    def rates(self) -> List[int]:
        """len(K) List[int] where k'th element gives the value of the rate coefficient for the k'th reaction"""
        return self._rates
    @property
    def species(self) -> List[str]:
        """len(N) List[str] where n'th element is the name of the n'th species"""
        return self._species
    @property
    def reaction_matrix(self) -> np.ndarray:
        """(K,N) array where [k,n] element gives the change in the n'th species for the k'th reaction"""
        return self._reaction_matrix
    @property
    def species_in_reaction(self):
        """len(K) List[List[int]] where k'th element gives the
        indices of `self.species` that are involved in the k'th reaction"""
        return self._species_in_reaction

    def _parse_reactions(self):
        reactions = []
        # deepcopy because update_rates() would otherwise change rates in self._original_config
        config_data = deepcopy(self._original_config)
        for k, (reaction, rate_list) in enumerate(config_data['reactions'].items()):
            reaction_string = reaction
            reactants = reaction.replace(' ', '').split('->')[0].split('+')
            products = reaction.replace(' ', '').split('->')[1].split('+')
            rate_name = rate_list[0]
            rate = rate_list[1]
            #positive_term =
            reactions.append(self.Reaction(
                k=k, reaction=reaction_string, reactants=reactants, products=products, rate_name=rate_name, rate=rate))
        return reactions

    def _parse_constraints(self):
        constraints = []
        config_data = deepcopy(self._original_config)
        for constraint in config_data['constraints']:
            if '<' in constraint:
                separator = '<='
            elif '>' in constraint:
                separator = '>='
            else:
                separator = '='
            species_involved = sorted(constraint.split(separator)[0].replace(' ', '').split('+'))
            value = int(constraint.split(separator)[1].replace(' ', ''))
            constraints.append(self.Constraint(
                constraint=constraint, species_involved=species_involved, separator=separator, value=value))
        return constraints

    def _set_species_vector(self):
        species_from_reactions = []
        for reaction in self._reactions:
            species_from_reactions += reaction.reactants + reaction.products
        while '0' in species_from_reactions:
            species_from_reactions.remove('0')
        # remove duplicates and sort
        species_from_reactions = sorted(list(set(species_from_reactions)))
        species_from_constraints = []
        for Constraint in self._constraints:
            species_from_constraints += Constraint.species_involved
        # this should be true or I can't build state space later
        assert(sorted(species_from_constraints) == species_from_reactions)
        return species_from_constraints

    def _set_reaction_matrix(self):
        K = len(self._reactions)
        N = len(self._species)
        reaction_matrix = np.zeros(shape=(K, N), dtype=int)
        for reaction in self._reactions:
            for n, species in enumerate(self._species):
                # if this species in both a product and reactant, net effect is 0
                if species in reaction.reactants:
                    reaction_matrix[reaction.k, n] += -1
                if species in reaction.products:
                    reaction_matrix[reaction.k, n] += 1
        return reaction_matrix

    def _set_species_in_reaction(self):
        K = len(self._reactions)
        N = len(self._species)
        indices = [[n for n in range(N) if self._reaction_matrix[k, n] < 0] for k in range(K)]
        return indices

    def print_propensities(self):
        """generates a readable list of the propensity of each reaction"""

        propensity_strings: List[str] = []
        for n_ids, rate in zip(self._species_in_reaction, self.rates):
            transition_rate = rate[0]
            for n in n_ids:
                transition_rate += f'c^{self._species[n]}'
            propensity_strings.append(transition_rate)

        return propensity_strings
