

class RefBinary:
    reactions_file = 'config_files/binary.yml'
    reactions = ['A -> B',
                 'B -> A']
    reactants = [['A'], ['B']]
    products = [['B'], ['A']]
    K = 2
    rate_names = ['k_AB', 'k_BA']
    rates = [1, 1]
    reaction_matrix = [[-1, 1], [1,-1]]
    species = ['A', 'B']
    N = 2
    species_in_reaction = [[0], [1]]
    constrained_species = [['A', 'B']]
    constraint_separators = ['=']
    constraint_values = [1]
    states = [[0,1], [1,0]]
    M = 2
    G_sparse = [-1, -1, 1, 1]
    G_dense = [[-1, 1], [1,-1]]
    k_to_G_map = {}


class Ref2IsoBinary:
    reactions_file = 'config_files/2_isolated_switches.yml'
    reactions = ['A1 -> A2',
                 'A2 -> A1',
                 'B1 -> B2',
                 'B2 -> B1']
    reactants = [['A1'], ['A2'], ['B1'], ['B2']]
    products = [['A2'], ['A1'], ['B2'], ['B1']]
    rate_names = ['k_A12', 'k_A21', 'k_B12', 'k_B21']
    rates = [1, 1, 1, 1]
    species = ['A1', 'A2', 'B1', 'B2']
    N = 4
    reaction_matrix = [[-1, 1, 0, 0],
                       [ 1,-1, 0, 0],
                       [ 0, 0,-1, 1],
                       [ 0, 0, 1,-1]]
    K = 4
    species_in_reaction = [[0], [1], [2], [3]]
    constrained_species = [['A1', 'A2'], ['B1', 'B2']]
    constraint_separators = ['=', '=']
    constraint_values = [1, 1]
    states = [[0,1,0,1],
              [0,1,1,0],
              [1,0,0,1],
              [1,0,1,0]]
    M = 4
    G_sparse = [-1, -1, 1, 1]
    G_dense = [[-1, 1], [1,-1]]
    k_to_G_map = {}



class RefBinary3N:
    ...

class RefAllostery:
    reactions_file = 'config_files/allostery.yml'
    reactions = ['A -> A*', 'A* -> A']
    reactants = [['A'], ['A*']]
    products = [['B'], ['A']]
    K = 12
    rate_names = ['k_A*+', 'k_A*-']
    #rates = [1, 1]
    #reaction_matrix = [[-1, 1], [1,-1]]
    species = ['A', 'A*', 'A*S', 'AS', 'B', 'BP', 'P', 'S']
    N = 8
    species_in_reaction = [[0], [1]]
    constrained_species = [['A', 'A*', 'A*S', 'AS'], ['B', 'BP'], ['P'], ['S']]
    constraint_separators = ['=', '=', '<=', '<=']
    constraint_values = [1, 1, 2, 2]
    #states = [[0,1,0,0,0,0,0,0],
    #          [1,0]]
    M = 0
    G_sparse = 0
    G_dense = [[-1, 1], [1,-1]]
    k_to_G_map = {}


