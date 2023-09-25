import math
import numpy as np
from .marginalization import get_PointMappings


def mutual_information(system, X, Y, base):

    N_timesteps = len(system.Trajectory.trajectory)
    PointMappings = get_PointMappings(system=system, X=X, Y=Y)

    mutual_information = np.zeros(shape=N_timesteps, dtype=np.float64)
    for ts in range(N_timesteps):
        mi_sum = 0
        P_t = system.Trajectory.trajectory[ts]
        for x, ids_x in PointMappings.X.items():
            for y, ids_y in PointMappings.Y.items():
                ids_xy = PointMappings.XY[x+y]
                p_xy = np.sum(P_t[ids_xy])
                if p_xy == 0:
                    # add zero to the sum if p_xy is 0
                    # need to do this because 0*np.log(0) returns an error
                    continue
                p_x = np.sum(P_t[ids_x])
                p_y = np.sum(P_t[ids_y])
                mi_sum += p_xy * math.log(p_xy / (p_x * p_y), base)
        mutual_information[ts] = mi_sum

    return mutual_information


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

