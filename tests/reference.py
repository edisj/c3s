

class RefBinary:
    reactions_file = 'config_files/binary.yml'
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
    k_to_G_map = {}


class Ref2IsoBinary:
    reactions_file = 'config_files/2_isolated_switches.yml'
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

    k_to_G_map = {}


class RefBinary3N:
    ...


class Ref4and2Switch:
    ...


class RefAllostery:
    reactions_file = 'config_files/allostery.yml'
    reactions = ['A -> A*',
                 'A* -> A',
                 'A + S -> AS',
                 'AS -> A + S',
                 'A* + S -> A*S',
                 'A*S -> A* + S',
                 'AS -> A*S',
                 'A*S -> AS',
                 'AS -> A + P',
                 'A*S -> A* + P',
                 'B + P -> BP',
                 'BP -> B + P',
                 '0 -> S',
                 'P -> 0']
    reactants = [['A'], ['A*']]
    products = [['A*'], ['A']]
    rate_names = ['k_A*+', 'k_A*-']
    #rates = [1, 1]
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
    G_dense = [[-1, 1], [1,-1]]
    k_to_G_map = {}'''


