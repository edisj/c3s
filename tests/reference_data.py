from c3s.config_files import (BINARY,
                              ISOLATED_2and2,
                              ISOLATED_4and2,
                              ALLOSTERY,
                              NO_ALLOSTERY)


"""TODO: explain this module"""


class RefBinary:
    reactions_file = BINARY
    reactions = ['A -> B',
                 'B -> A']
    reactants = [['A'], ['B']]
    products = [['B'], ['A']]
    rate_names = ['k_AB', 'k_BA']
    rates = [1, 1]
    species = ['A', 'B']
    N = 2
    reaction_matrix = [[-1, 1],
                       [ 1,-1]]
    K = 2
    species_in_reaction = [[0], [1]]
    constrained_species = [['A', 'B']]
    constraint_separators = ['=']
    constraint_values = [1]
    states = [[0,1],
              [1,0]]
    M = 2
    G_dense = [[-1, 1],
               [ 1,-1]]
    G_sparse = [-1, -1, 1, 1]
    updated_rates = {'k_AB': 2, 'k_BA': 3}
    G_dense_updated = [[-3, 2],
                       [ 3, -2]]
    G_sparse_updated = [-3, -2, 3, 2]


class Ref2and2Iso:
    reactions_file = ISOLATED_2and2
    reactions = ['A -> B',
                 'B -> A',
                 'X -> Y',
                 'Y -> X']
    reactants = [['A'], ['B'], ['X'], ['Y']]
    products = [['B'], ['A'], ['Y'], ['X']]
    rate_names = ['k_AB', 'k_BA', 'k_XY', 'k_YX']
    rates = [1, 2, 3, 4]
    species = ['A', 'B', 'X', 'Y']
    N = 4
    reaction_matrix = [[-1, 1, 0, 0],
                       [ 1,-1, 0, 0],
                       [ 0, 0,-1, 1],
                       [ 0, 0, 1,-1]]
    K = 4
    species_in_reaction = [[0], [1], [2], [3]]
    constrained_species = [['A', 'B'], ['X', 'Y']]
    constraint_separators = ['=', '=']
    constraint_values = [1, 1]
    states = [[0,1,0,1],
              [0,1,1,0],
              [1,0,0,1],
              [1,0,1,0]]
    M = 4
    G_dense = [[-6, 3, 1, 0],
               [ 4,-5, 0, 1],
               [ 2, 0,-5, 3],
               [ 0, 2, 4,-4]]
    G_sparse = [-6, -5, -5, -4, 2, 4, 2, 3, 1, 4, 1, 3]


class Ref4and2Iso:
    reactions_file = ISOLATED_4and2
    reactions = ['A -> B',
                 'B -> A',
                 'B -> C',
                 'C -> B',
                 'C -> D',
                 'D -> C',
                 'D -> A',
                 'A -> D',
                 'X -> Y',
                 'Y -> X']
    reactants = [['A'], ['B'], ['B'], ['C'], ['C'], ['D'], ['D'], ['A'], ['X'], ['Y']]
    products = [['B'], ['A'], ['C'], ['B'], ['D'], ['C'], ['A'], ['D'], ['Y'], ['X']]
    rate_names = ['k_AB', 'k_BA', 'k_BC', 'k_CB', 'k_CD', 'k_DC', 'k_DA', 'k_AD', 'k_XY', 'k_YX']
    rates = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
    species = ['A', 'B', 'C', 'D', 'X', 'Y']
    N = 6
    reaction_matrix = [[-1, 1, 0, 0, 0, 0],
                       [ 1,-1, 0, 0, 0, 0],
                       [ 0,-1, 1, 0, 0, 0],
                       [ 0, 1,-1, 0, 0, 0],
                       [ 0, 0,-1, 1, 0, 0],
                       [ 0, 0, 1,-1, 0, 0],
                       [ 1, 0, 0,-1, 0, 0],
                       [-1, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0,-1, 1],
                       [ 0, 0, 0, 0, 1,-1]]
    K = 10
    species_in_reaction = [[0], [1], [1], [2], [2], [3], [3], [0], [4], [5]]
    constrained_species = [['A', 'B', 'C', 'D'], ['X', 'Y']]
    constraint_separators = ['=', '=']
    constraint_values = [1, 1]
    states = [[0,0,0,1,0,1],
              [0,0,0,1,1,0],
              [0,0,1,0,0,1],
              [0,0,1,0,1,0],
              [0,1,0,0,0,1],
              [0,1,0,0,1,0],
              [1,0,0,0,0,1],
              [1,0,0,0,1,0]]
    M = 8
    G_dense = [[-4, 2, 1, 0, 0, 0, 1, 0],
               [ 2,-4, 0, 1, 0, 0, 0, 1],
               [ 1, 0,-4, 2, 1, 0, 0, 0],
               [ 0, 1, 2,-4, 0, 1, 0, 0],
               [ 0, 0, 1, 0,-4, 2, 1, 0],
               [ 0, 0, 0, 1, 2,-4, 0, 1],
               [ 1, 0, 0, 0, 1, 0,-4, 2],
               [ 0, 1, 0, 0, 0, 1, 2,-4]]
    G_sparse = [-4, -4, -4, -4, -4, -4, -4, -4, 1, 1, 2, 1, 1, 2, 1, 1, 2,
                1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]


class RefAllostery:
    reactions_file = ALLOSTERY
    reactions = ['A -> A*',
                 'A* -> A',
                 'A* + S -> A*S',
                 'A*S -> A* + S',
                 'A*S -> AS',
                 'AS -> A*S',
                 'AS -> A + S',
                 'A + S -> AS',
                 'AS -> A + P',
                 'A*S -> A* + P',
                 'B + P -> BP',
                 'BP -> B + P',
                 '0 -> S',
                 'P -> 0']
    reactants = [['A'], ['A*'], ['A*', 'S'], ['A*S'], ['A*S'], ['AS'], ['AS'], ['A', 'S'], ['AS'], ['A*S'], ['B','P'], ['BP'], ['0'], ['P']]
    products = [['A*'], ['A'], ['A*S'], ['A*', 'S'], ['AS'], ['A*S'], ['A', 'S'], ['AS'], ['A','P'], ['A*', 'P'], ['BP'], ['B','P'], ['S'], ['0']]
    rate_names = ['k_A*+', 'k_A*-', 'k_S*+', 'k_A*S-', 'k_A*S+', 'k_S+', 'k_cat', 'k_cat*', 'k_BP+', 'k_BP-', 'beta_S', 'gamma_P']
    rates = [2, 1, 4, 1, 1, 2, 1, 2, 1, 10, 2, 1, 1, 1]
    species = ['A', 'A*', 'A*S', 'AS', 'B', 'BP', 'S', 'P']
    N = 8
    '''reaction_matrix = [[-1, 1, 0, 0, 0, 0, 0, 0],
                       [ 1,-1, 0, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 1, 0, 0,-1, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 0, 0,-1]]
    K = 14
    species_in_reaction = [[0], [1]]
    constrained_species = [['A', 'A*', 'A*S', 'AS'], ['B', 'BP'], ['S'], ['P']]
    constraint_separators = ['=', '=', '<=', '<=']
    constraint_values = [1, 1, 1, 1]
    states = [[0,0,0,1,0,1,0,0],
              [0,0,0,1,0,1,0,1],
              [0,0,1,0,0,0,0,0],
              [0,1,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0]]
    M = 0
    G_sparse = 0
    G_dense = [[-1, 1], [1,-1]]'''


