import numpy as np

binary = {
    'N': 2,
    'species': ['A', 'B'],
    'K': 2,
    'rates': [1, 1],
    'rate_names': ['k_AB', 'k_BA'],
    'reaction_matrix': np.array([[-1,  1],
                                 [ 1, -1]]),
    'states': np.array([[0, 1],
                        [1, 0]]),
    'G': 0}

binary3N = {}

binary10N = {}

ternary = {}

two_binary_switch = {
    'N': 0,
    'species': 0,
    'constrained_species': 0,
    'constraint_values': 0,

    'K': 0,
    'rates': 0,
    'rate_names': 0,
    'states': 0,
    'G': 0,

}