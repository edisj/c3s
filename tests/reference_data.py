from c3s.config_files import (BINARY,
                              ISOLATED_2and2,
                              ISOLATED_4and2,
                              ALLOSTERY,
                              NO_ALLOSTERY)
import numpy as np
from math import factorial

"""TODO: explain this module"""

class RefBINARY:
    config = BINARY
    reactions = ['A -> B', 'B -> A']
    reactants = [['A'], ['B']]
    products = [['B'], ['A']]
    rate_names = ['k_AB', 'k_BA']
    rates = [1, 1]
    species = ['A', 'B']
    M=2
    N=2
    K=2
    initial_populations = {'A':1}
    initial_state = [1,0]
    initial_state_index = 1
    reaction_matrix = [[-1, 1],
                        [1,-1]]
    species_in_reaction = [[0], [1]]
    constrained_species = [['A', 'B']]
    constraint_separators = ['=']
    constraint_values = [1]
    states = [[0, 1],
              [1, 0]]
    G_dense = [[-1, 1],
               [ 1,-1]]
    G_sparse = [-1, -1, 1, 1]
    species_subset = ['B']
    point_map_keys = [(0,), (1,)]
    point_map_ids = [[1], [0]]
    updated_rates = {'k_AB': 2, 'k_BA': 3}
    G_dense_updated = [[-3, 2],
                       [ 3, -2]]
    G_sparse_updated = [-3, -2, 3, 2]

    A_marginalized_tuples = [(0,), (1,)]
    B_marginalized_tuples = [(1,), (0,)]
    X_set = ['A', ['A'], 'B']
    Y_set = ['B', ['B'], 'A']
    ss_mutual_informations = [1.0]*3


class Ref2and2Iso:
    config = ISOLATED_2and2
    reactions=['A -> B',
               'B -> A',
               'X -> Y',
               'Y -> X']
    reactants = [['A'], ['B'], ['X'], ['Y']]
    products = [['B'], ['A'], ['Y'], ['X']]
    rate_names = ['k_AB', 'k_BA', 'k_XY', 'k_YX']
    rates = [1, 2, 3, 4]
    species = ['A', 'B', 'X', 'Y']
    initial_populations={'A': 1, 'X': 1}
    initial_state=[1, 0, 1, 0]
    initial_state_index = 3
    M = 4
    N = 4
    K = 4
    reaction_matrix = [[-1, 1, 0, 0],
                       [1, -1, 0, 0],
                       [0, 0, -1, 1],
                       [0, 0, 1, -1]]
    species_in_reaction = [[0], [1], [2], [3]]
    constrained_species = [['A', 'B'], ['X', 'Y']]
    constraint_separators = ['=', '=']
    constraint_values = [1, 1]
    states=[[0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0]]
    G_dense=[[-6, 3, 1, 0],
             [4, -5, 0, 1],
             [2, 0, -5, 3],
             [0, 2, 4, -4]]
    G_sparse = [-6, -5, -5, -4, 2, 4, 2, 3, 1, 4, 1, 3]

    species_subset = ['X', 'B']
    point_map_keys = [(0,0), (0,1), (1,0), (1,1)]
    point_map_ids = [[2], [0], [3], [1]]

    updated_rates = {'k_AB': 10, 'k_XY': 20}
    G_dense_updated = [[-6, 20,  10,   0],
                       [ 4,-22,   0,  10],
                       [ 2,  0, -14,  20],
                       [ 0,  2,   4, -30]]
    G_sparse_updated = [-6, -22, -14, -30, 2, 4, 2, 20, 10, 4, 10, 20]

    X_set = [['A', 'B']]
    Y_set = [['X', 'Y']]
    ss_mutual_informations = [0.0]


class Ref4and2Iso:
    config = ISOLATED_4and2
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
    M=8
    K=10
    N = 6
    species = ['A', 'B', 'C', 'D', 'X', 'Y']
    initial_populations={'A': 1, 'Y': 1}
    initial_state=[1, 0, 0, 0, 0, 1]
    initial_state_index=6
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
    '''updated_rates = {'k_AB': 10, 'k_XY': 20}
    G_dense_updated = [[-4, 2, 1, 0, 0, 0, 1, 0],
                       [ 2,-4, 0, 1, 0, 0, 0, 1],
                       [ 1, 0,-4, 2, 1, 0, 0, 0],
                       [ 0, 1, 2,-4, 0, 1, 0, 0],
                       [ 0, 0, 1, 0,-4, 2, 1, 0],
                       [ 0, 0, 0, 1, 2,-4, 0, 1],
                       [ 1, 0, 0, 0, 1, 0,-4, 2],
                       [ 0, 1, 0, 0, 0, 1, 2,-4]]
    G_sparse_updated =  [-4, -4, -4, -4, -4, -4, -4, -4, 1, 1, 2, 1, 1, 2, 1, 1, 2,
                          1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]

    species_subset = ['B', 'X']
    point_maps_keys = [(0,0), (0,1), (1,0), (1,1)]
    point_map_ids = [[0,2,6], [1,3,7], [4], [5]]'''

    X_set = [['A', 'B', 'C', 'D'], 'X']
    Y_set = [['X', 'Y'], 'Y']
    ss_mutual_informations = [0.0, 1.0]


def _build_G_manually_no_allostery():

    maxA = 1
    maxB = 1
    maxS = 2
    maxP = 2
    Na = factorial(maxA + 2 - 1) / (factorial(maxA) * factorial(2 - 1))
    Nb = factorial(maxB + 2 - 1) / (factorial(maxB) * factorial(2 - 1))
    Ns = maxS + 1
    Np = maxP + 1
    rates = [1, 2, 3, 4, 5, 0.5, 0.1]

    M = Na * Nb * Ns * Np
    M = int(M)
    Npsb = int(Ns * Np * Nb)
    Nps = int(Np * Ns)

    G = np.zeros((M, M))
    G_sparse = []
    for j in range(M):

        nS = (j // Np) % Ns
        nP = j % Np

        # A
        if j // Npsb == 1:
            # A + S -> AS 1
            if nS != 0:
                i = j - Npsb - Np
                G[i, j] = nS * rates[0]
                G_sparse.append(nS * rates[0])
        # AS
        if j // Npsb == 0:
            # AS -> A + S
            if nS != maxS:
                i = j + Npsb + Np
                G[i, j] = rates[1]
                G_sparse.append(rates[1])
            # AS -> A + P
            if nP != maxP:
                i = j + Npsb + 1
                G[i, j] = rates[2]
                G_sparse.append(rates[2])
        # B
        if (j // Nps) % 2 == 1:
            # B + P -> BP
            if nP != 0:
                i = j - Nps - 1
                G[i, j] = nP * rates[3]
                G_sparse.append(nP * rates[3])
        # BP
        if (j // Nps) % 2 == 0:
            # BP -> B + P
            if nP != maxP:
                i = j + Nps + 1
                G[i, j] = rates[4]
                G_sparse.append(rates[4])
        # 0 -> S
        if nS != maxS:
            i = j + Np
            G[i, j] = rates[5]
            G_sparse.append(rates[5])
        # P -> 0
        if nP != 0:
            i = j - 1
            G[i, j] = nP * rates[6]
            G_sparse.append(nP * rates[6])

    for m in range(M):
        G[m, m] = -np.sum(G[:, m])
    G_sparse = G.diagonal().tolist() + G_sparse

    return G, G_sparse

class RefNoAllostery:
    config = NO_ALLOSTERY
    reactions=['A + S -> AS',
               'AS -> A + S',
               'AS -> A + P',
               'B + P -> BP',
               'BP -> B + P',
               '0 -> S',
               'P -> 0']
    reactants=[['A', 'S'], ['AS'], ['AS'], ['B', 'P'], ['BP'], ['0'], ['P']]
    products=[['AS'], ['A', 'S'], ['A', 'P'], ['BP'], ['B', 'P'], ['S'], ['0']]
    rate_names=['kon', 'koff', 'kcat', 'kPon', 'kPoff', 'beta', 'gamma']
    rates=[1,2,3,4,5,0.5,0.1]
    N=6
    species=['A', 'AS', 'B', 'BP', 'S', 'P']
    initial_populations={'AS': 1, 'BP': 1}
    initial_state=[0, 1, 0, 1, 0, 0]
    initial_state_index=0
    M=2*2*3*3
    K=7
    reaction_matrix=[[-1,  1,  0,  0, -1,  0],
                     [ 1, -1,  0,  0,  1,  0],
                     [ 1, -1,  0,  0,  0,  1],
                     [ 0,  0, -1,  1,  0, -1],
                     [ 0,  0,  1, -1,  0,  1],
                     [ 0,  0,  0,  0,  1,  0],
                     [ 0,  0,  0,  0,  0, -1]]
    species_in_reaction=[[0, 4], [1], [1], [2,5], [3], [], [5]]
    constrained_species=[['A', 'AS'], ['B', 'BP'], ['S'], ['P']]
    constraint_separators=['=', '=', '<=', '<=']
    constraint_values=[1, 1, 2, 2]
    states=[[0, 1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0, 1],
           [0, 1, 0, 1, 0, 2],
           [0, 1, 0, 1, 1, 0],
           [0, 1, 0, 1, 1, 1],
           [0, 1, 0, 1, 1, 2],
           [0, 1, 0, 1, 2, 0],
           [0, 1, 0, 1, 2, 1],
           [0, 1, 0, 1, 2, 2],
           [0, 1, 1, 0, 0, 0],
           [0, 1, 1, 0, 0, 1],
           [0, 1, 1, 0, 0, 2],
           [0, 1, 1, 0, 1, 0],
           [0, 1, 1, 0, 1, 1],
           [0, 1, 1, 0, 1, 2],
           [0, 1, 1, 0, 2, 0],
           [0, 1, 1, 0, 2, 1],
           [0, 1, 1, 0, 2, 2],
           [1, 0, 0, 1, 0, 0],
           [1, 0, 0, 1, 0, 1],
           [1, 0, 0, 1, 0, 2],
           [1, 0, 0, 1, 1, 0],
           [1, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, 1, 2],
           [1, 0, 0, 1, 2, 0],
           [1, 0, 0, 1, 2, 1],
           [1, 0, 0, 1, 2, 2],
           [1, 0, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 1],
           [1, 0, 1, 0, 0, 2],
           [1, 0, 1, 0, 1, 0],
           [1, 0, 1, 0, 1, 1],
           [1, 0, 1, 0, 1, 2],
           [1, 0, 1, 0, 2, 0],
           [1, 0, 1, 0, 2, 1],
           [1, 0, 1, 0, 2, 2]]
    G_dense=_build_G_manually_no_allostery()[0]
    G_sparse=_build_G_manually_no_allostery()[1]


class RefAllosteric2State:
    config = ALLOSTERY
    reactions = ['A -> A*',
                 'A* -> A',
                 'AS -> A*S',
                 'A*S -> AS',
                 'A + S -> AS',
                 'AS -> A + S',
                 'A* + S -> A*S',
                 'A*S -> A* + S',
                 'AS -> A + P',
                 'A*S -> A* + P',
                 'B + P -> BP',
                 'BP -> B + P',
                 '0 -> S',
                 'P -> 0']
    reactants = [['A'], ['A*'], ['AS'], ['A*S'], ['A', 'S'], ['AS'], ['A*', 'S'], ['A*S'], ['AS'], ['A*S'],
                 ['B','P'], ['BP'], ['0'], ['P']]
    products = [['A*'], ['A'], ['A*S'], ['AS'], ['AS'], ['A', 'S'], ['A*S'], ['A*', 'S'], ['A','P'], ['A*', 'P'],
                ['BP'], ['B','P'], ['S'], ['0']]
    rate_names = ['kA', 'k*A', 'kAS', 'k*AS', 'kon', 'koff', 'k*on', 'k*off', 'kcat', 'k*cat',
                  'kP_on', 'kP_off', 'beta', 'gamma']
    rates = [1, 2, 2, 1, 2, 1, 2, 1, 1, 10, 1, 2, 1, 1]
    N = 8
    species = ['A', 'A*', 'A*S', 'AS', 'B', 'BP', 'S', 'P']
    initial_populations={'AS': 1, 'BP': 1}
    initial_state=[0, 0, 0, 1, 0, 1, 0, 0]
    initial_state_index=0
    M = 4*2*2*2
    K = 14
    reaction_matrix = [[-1,  1,  0,  0,  0,  0,  0,  0],
                       [ 1, -1,  0,  0,  0,  0,  0,  0],
                       [ 0,  0,  1, -1,  0,  0,  0,  0],
                       [ 0,  0, -1,  1,  0,  0,  0,  0],
                       [-1,  0,  0,  1,  0,  0, -1,  0],
                       [ 1,  0,  0, -1,  0,  0,  1,  0],
                       [ 0, -1,  1,  0,  0,  0, -1,  0],
                       [ 0,  1, -1,  0,  0,  0,  1,  0],
                       [ 1,  0,  0, -1,  0,  0,  0,  1],
                       [ 0,  1, -1,  0,  0,  0,  0,  1],
                       [ 0,  0,  0,  0, -1,  1,  0, -1],
                       [ 0,  0,  0,  0,  1, -1,  0,  1],
                       [ 0,  0,  0,  0,  0,  0,  1,  0],
                       [ 0,  0,  0,  0,  0,  0,  0, -1]]
    species_in_reaction = [[0], [1], [3], [2], [0,6], [3], [1,6], [2], [3], [2], [4,7], [5], [], [7]]
    constrained_species = [['A', 'A*', 'A*S', 'AS'], ['B', 'BP'], ['S'], ['P']]
    constraint_separators = ['=', '=', '<=', '<=']
    constraint_values = [1, 1, 1, 1]
    states = [[0,0,0,1,0,1,0,0],
              [0,0,0,1,0,1,0,1],
              [0,0,1,0,0,0,0,0],
              [0,1,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0]]
    G_sparse = 0
    G_dense = [[-1, 1], [1,-1]]
