import numpy as np
from math import log
from typing import  Dict, Tuple
from collections import namedtuple


class Calculate:
    def __init__(self, system):
        self.system = system
        self.Data = namedtuple("Data", ['data', 'X', 'Y', 'base'])

    def mutual_information(self, X, Y, base=2):
        '''Mutual information over X and Y
        I(X,Y) = sum_x,y p(x,y)log[ p(x,y) / (p(x)p(y)) ]
        '''
        X = [X] if isinstance(X, str) else X
        Y = [Y] if isinstance(Y, str) else Y
        X_traj = self.marginalize_trajectory(X)
        Y_traj = self.marginalize_trajectory(Y)
        XY_traj = self.marginalize_trajectory(X + Y)

        mutual_information = []
        for ts in range(self.system.Trajectory.N_timesteps):
            _sum = 0.0
            for x, Px in X_traj.items():
                for y, Py in Y_traj.items():
                    try:
                        Pxy = XY_traj[x + y]
                        p_xy = Pxy[ts]
                    except KeyError:
                        # x+y was not in state space
                        continue
                    if p_xy == 0:
                        # 0*log(0) -> 0
                        continue
                    p_x, p_y = Px[ts], Py[ts]
                    _sum += p_xy*log( (p_xy / (p_x*p_y) ), base)
            mutual_information.append(_sum)

        return self.Data(data=np.asarray(mutual_information), X=X, Y=Y, base=base)

    def entropy(self, X, base=2):
        '''Shannon entropy
        H(X) = -sum_x p(x)log(p(x))
        '''
        X = [X] if isinstance(X, str) else X
        X_traj = self.marginalize_trajectory(X)
        Hx = []
        for ts in range(self.system.Trajectory.N_timesteps):
            _sum = 0.0
            for Px in X_traj.values():
                p_x = Px[ts]
                if p_x == 0:
                    continue
                _sum += -p_x*log(p_x, base)
            Hx.append(_sum)

        return self.Data(data=np.asarray(Hx), X=X, Y=None, base=base)

    def conditional_entropy(self, X, Y, base=2):
        """
        H(X|Y) = -sum_x,y p(x,y)log( p(x,y)/p(y) )
        """
        X = [X] if isinstance(X, str) else X
        Y = [Y] if isinstance(Y, str) else Y
        X_traj = self.marginalize_trajectory(X)
        Y_traj = self.marginalize_trajectory(Y)
        XY_traj = self.marginalize_trajectory(X + Y)
        H_X_given_Y = []
        for ts in range(self.system.Trajectory.N_timesteps):
            _sum = 0.0
            for x, Px in X_traj.items():
                for y, Py in Y_traj.items():
                    try:
                        Pxy = XY_traj[x + y]
                    except KeyError:
                        # x+y was not in state space
                        continue
                    p_xy = Pxy[ts]
                    if p_xy == 0:
                        # 0 * log(0) -> 0
                        continue
                    p_y = Py[ts]
                    _sum += -p_xy*log((p_xy/p_y), base)
            H_X_given_Y.append(_sum)

        return self.Data(data=np.asarray(np.asarray(H_X_given_Y)), X=X, Y=Y, base=base)

    def marginalize_trajectory(self, species_subset) -> Dict[Tuple, np.ndarray]:
        point_mappings = self._get_point_mappings(species_subset)
        marginalized_trajectory = {state: np.asarray([P_t[ids].sum() for P_t in self.system.trajectory])
                                   for state, ids in point_mappings.items()}
        return marginalized_trajectory

    def avg_copy_number(self, species) -> np.ndarray:
        """calculates the average copy number for a single chemical species over the probability trajectory"""
        marginalized_trajectory = self.marginalize_trajectory(species)
        avg_copy_number = np.asarray([sum([c[0] * P_c[ts] for c, P_c in marginalized_trajectory.items()])
                                      for ts in range(len(self.system.trajectory))])
        return avg_copy_number

    def _get_point_mappings(self, species_subset) -> Dict[Tuple, np.ndarray]:
        """Maps each microstate vector in `system.states` to their marginal probability distribution
        domain point for the specified molecular species if you were to sum over all other species != `species`.

        e.g. find where each (c_A, c_B, c_C, c_D) point in p(c_A, c_B, c_C, c_D) maps to in p(c_A, c_B) if the set
        of (c_A, c_B, c_C, c_D) points were flattened into a rank-1 array where c_C and c_D are summed over.

        """
        if self.system._Trajectory is None:
            raise AttributeError("no data in `system.trajectory`")
        species_subset = [species_subset] if isinstance(species_subset, str) else species_subset
        ids = [self.system.species.index(species) for species in species_subset]
        subspace = np.unique(self.system.states[:, ids], axis=0)
        point_mappings = {
            tuple(vector) : np.where(np.all(self.system.states[:, ids] == vector, axis=1))[0] for vector in subspace}
        return point_mappings


"""def generate_analytic_MI_function(self, X, Y, base=2):
    Deltas = self._fix_species_and_get_Deltas(X, Y)

    indices_for_each_term = [
        [np.argwhere(Deltas.xy[x + y]).T.tolist()[0], np.argwhere(Dx).T.tolist()[0], np.argwhere(Dy).T.tolist()[0]]
        for x, Dx in Deltas.x.items() for y, Dy in Deltas.y.items() if x + y in Deltas.xy]

    def analytic_function(P, base=base):
        term_values = []
        for ids in indices_for_each_term:
            Pxy = sum([P[i] for i in ids[0]])
            Px = sum([P[i] for i in ids[1]])
            Py = sum([P[i] for i in ids[2]])
            if Pxy == 0:
                this_term = 0.0
            else:
                this_term = Pxy * math.log(Pxy / (Px * Py), base)
            term_values.append(this_term)
        return sum(term_values)

    return analytic_function


def _get_analytic_string(self, X, Y):
    Deltas = self._fix_species_and_get_Deltas(X, Y)

    indices_for_each_term = [
        [np.argwhere(Deltas.xy[x + y]).T.tolist()[0], np.argwhere(Dx).T.tolist()[0], np.argwhere(Dy).T.tolist()[0]]
        for x, Dx in Deltas.x.items() for y, Dy in Deltas.y.items() if x + y in Deltas.xy]
    string_per_term = []
    for ids in indices_for_each_term:
        Pxy = [f"p{i + 1}" for i in ids[0]]
        Px = [f"p{i + 1}" for i in ids[1]]
        Py = [f"p{i + 1}" for i in ids[2]]

        term = f"{Pxy}log({Pxy}/({Px}{Py}))"
        string_per_term.append(term)

    return string_per_term"""
