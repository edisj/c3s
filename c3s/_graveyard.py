import numpy as np
import math
from collections import namedtuple


# code that I'm temporarily keeping for reference, will delete later...

def print_constitutive_states(self):
        """generates a convenient readable list of the constitutive states"""

        constitutive_states_strings = []
        for state in self._constitutive_states:
            word = []
            for population_number, species in zip(state, self.species):
                word.append(f'{population_number}{species}')
            constitutive_states_strings.append(word)

        return constitutive_states_strings


def print_generator_matrix(self):
    """generates a convenient readable generator matrix with string names"""

    M, K = self.M, self.K
    readable_G = [['0' for _ in range(M)] for _ in range(M)]
    propensity_strings = self.reaction_network.print_propensities()
    for k in range(K):
        for idx in self._nonzero_G_elements[k]:
            i,j = idx
            readable_G[i][j] = propensity_strings[k]
    for j in range(M):
        diagonal = '-('
        for i in range(M):
            if readable_G[i][j] != '0':
                diagonal += f'{readable_G[i][j]} + '
        readable_G[j][j] = diagonal[:-3] + ')'

    return readable_G


def __build_constitutive_states_OLD(self):
    """this is old and slow version, won't run at the moment"""

    constitutive_states = [self._initial_state]
    population_limits = [
        (self.species.index(species), max_count)
        for species, max_count in self._max_populations.items()
    ] if self._max_populations else False

    self._nonzero_G_elements = {k: [] for k in range(len(self.reaction_network.reaction_matrix))}

    # newly_added keeps track of the most recently accepted states
    newly_added_states = [np.array(self._initial_state)]
    while True:
        accepted_candidate_states = []
        for state in newly_added_states:
            i = int(np.argwhere(np.all(constitutive_states == state, axis=1)))
            # the idea here is that for each of the recently added states,
            # we iterate through each reaction to see if a transition is possible
            for k, reaction in enumerate(self.reaction_network.reaction_matrix):
                # gives a boolean array for which reactants are required
                reactants_required = np.argwhere(reaction < 0).T
                reactants_available = state > 0
                # true if this candidate state has all of the reactants available for the reaction
                if np.all(reactants_available[reactants_required]):
                    # apply the reaction and add the new state into our list of constitutive
                    # states only if it is a new state that has not been previously visited
                    new_candidate_state = state + reaction
                    is_actually_new_state = list(new_candidate_state) not in constitutive_states
                    does_not_exceed_max_population = all([new_candidate_state[i] <= max_count for i, max_count
                                                          in population_limits]) if population_limits else True
                    if does_not_exceed_max_population:
                        if is_actually_new_state:
                            j = len(constitutive_states)
                            accepted_candidate_states.append(new_candidate_state)
                            constitutive_states.append(list(new_candidate_state))
                        else:
                            j = int(np.argwhere(np.all(constitutive_states == new_candidate_state, axis=1)))
                        self._nonzero_G_elements[k].append((j,i))

        # replace the old set of new states with new batch
        newly_added_states = accepted_candidate_states
        # once we reach the point where no new states are accessible we terminate
        if not newly_added_states:
            break

    self._constitutive_states = np.array(constitutive_states, dtype=np.int32)


def __build_generator_matrix_OLD(self):
    """old and slow version, probably won't run anymore"""

    M = self.M
    G = np.zeros(shape=(M,M), dtype=float)
    for k, value in self._nonzero_G_elements.items():
        for idx in value:
            i,j = idx
            # the indices of the species involved in the reaction
            n_ids = self.reaction_network.species_in_reaction[k]
            # h is the combinatorial factor for number of reactions attempting to fire
            # At the moment this assumes maximum stoichiometric coefficient of 1
            # TODO: generalize h for any coefficient
            state_j = self._constitutive_states[j]
            h = np.prod([state_j[n] for n in n_ids])
            rate = self.reaction_network.reactions[k].rate
            reaction_propensity = h * rate
            G[i,j] = reaction_propensity
    for i in range(M):
        # fix the diagonal to be the negative sum of the column
        G[i,i] = -np.sum(G[:,i])
    return G


def _calculate_mutual_information_diagonal(self, Deltas, base):
    """"""

    N_timesteps = len(self.trajectory)
    mut_inf = np.zeros(shape=N_timesteps, dtype=np.float64)

    mi_terms = {}
    for ts in range(N_timesteps):
        P = self.trajectory[ts]
        mi_sum = 0
        for x, Dx in Deltas.x.items():
            for y, Dy in Deltas.y.items():
                xy = x + y
                if xy not in Deltas.xy:
                    # skip cases where the concatenated coordinate tuples
                    # were never in the joint distribution to begin with
                    continue
                Dxy = Deltas.xy[xy]
                p_xy = np.dot(P, Dxy)
                if p_xy == 0:
                    # add zero to the sum if p_xy is 0
                    # need to do this because 0*np.log(0) returns an error
                    continue
                p_x = np.dot(P, Dx)
                p_y = np.dot(P, Dy)
                this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
                mi_sum += this_term
                mi_terms[xy] = this_term

        mut_inf[ts] = mi_sum

    return mut_inf, mi_terms


def _calculate_mutual_information_matrix(self, Deltas, base):
    """calculates the pairwise instantaneous mutual information between
    every timepoint in the trajectory"""

    N_timesteps = len(self.trajectory)
    global_mi_matrix = np.zeros(shape=(N_timesteps, N_timesteps), dtype=np.float64)
    local_dts = [self._dt * ts for ts in range(N_timesteps)]
    # using a generator expression to save on memory here
    local_Qs = (expm(self.G*dt) for dt in local_dts)
    for top_col_loc, Q in zip(range(N_timesteps), local_Qs):
        if top_col_loc == 0:
            diagonal = self._calculate_mutual_information_diagonal(Deltas, base)
            np.fill_diagonal(global_mi_matrix, diagonal)
            continue
        # first Q is upper triangle, second is for lower triangle
        tildeQ_matrices = {(x,y): (Q*np.outer(Dy, Dx), Q*np.outer(Dx, Dy))
                                  for x, Dx in Deltas.x.items() for y, Dy in Deltas.y.items()}
        # here we calculate the mutual information along the offdiagonal
        # so as to avoid repeatedly computing the matrix exponential
        i =  0
        j = top_col_loc
        while j < N_timesteps:    # the edge of the matrix
            upper, lower = self._calculate_matrix_elements(i, j, Deltas, tildeQ_matrices, base)
            global_mi_matrix[i,j] = upper
            global_mi_matrix[j,i] = lower
            i += 1
            j += 1

    return global_mi_matrix


def _calculate_matrix_elements(self, i, j, Deltas, tildeQ_matrices, base):
    """calculates every matrix element for symmetric off diagonals"""

    # here I assume j is always at the later timepoint
    Pi = self.trajectory[i]
    Pj = self.trajectory[j]

    mi_sum_upper = 0
    for x, Dx in Deltas.x.items():
        for y, Dy in Deltas.y.items():
            tildeQ = tildeQ_matrices[(x,y)][0]
            p_xy= np.dot(np.dot(tildeQ, Pi), Dy)
            if p_xy == 0:
                # add zero to the sum if p_xy is 0
                # need to do this because 0*np.log(0) returns an error
                continue
            p_x = np.dot(Pi, Dx)
            p_y = np.dot(Pj, Dy)
            this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
            mi_sum_upper += this_term
    mi_sum_lower = 0
    for x, Dx in Deltas.x.items():
        for y, Dy in Deltas.y.items():
            tildeQ = tildeQ_matrices[(x,y)][1]
            p_xy= np.dot(np.dot(tildeQ, Pi), Dx)
            if p_xy == 0:
                # add zero to the sum if p_xy is 0
                # need to do this because 0*np.log(0) returns an error
                continue
            p_x = np.dot(Pj, Dx)
            p_y = np.dot(Pi, Dy)
            this_term = p_xy * math.log(p_xy / (p_x * p_y), base)
            mi_sum_lower += this_term

    return mi_sum_upper, mi_sum_lower


def _fix_species_and_get_Deltas(self, X, Y):
    if isinstance(X, str):
        X = [X]
    if isinstance(Y, str):
        Y = [Y]
    #X = sorted(X)
    #Y = sorted(Y)
    #if X + Y != sorted(X + Y):
    #    X, Y = Y, X

    DeltaTuple = namedtuple("Deltas", "x y xy")
    Deltas = DeltaTuple(
        self._get_Delta_vectors(X),
        self._get_Delta_vectors(Y),
        self._get_Delta_vectors(X + Y))

    return Deltas


def _get_Delta_vectors(self, molecules):
    if isinstance(molecules, str):
        molecules = [molecules]
    # indices that correspond to the selected molecular species in the ordered species list
    n_ids = [self.species.index(n) for n in molecules]
    truncated_points = np.array(self._constitutive_states)[:, n_ids]

    Delta_vectors = {}

    curr_state_id = 0
    states_accounted_for = []
    # begin mapping points until every state in the microstate space is accounted for
    while len(states_accounted_for) < len(truncated_points):
        # find the indices of degenerate domain points starting with the first state
        # and skipping iterations if that point has been accounted for
        if curr_state_id in states_accounted_for:
            curr_state_id += 1
            continue
        curr_state = truncated_points[curr_state_id]
        Dv = np.all(truncated_points == curr_state, axis=1).astype(np.float64)
        Delta_vectors[tuple(curr_state)] = Dv
        # keep track of which states we have accounted for
        states_accounted_for += np.argwhere(Dv == 1).transpose().tolist()
        curr_state_id += 1

    return Delta_vectors
