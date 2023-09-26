import math
import numpy as np
from .marginalization import get_point_mappings


def calculate_mutual_information(X, Y, system, base=2):
    x_map = get_point_mappings(system, X)
    y_map = get_point_mappings(system, Y)
    xy_map = get_point_mappings(system, X+Y)

    mutual_information = np.asarray(
        [sum([_summand(P[x_ids].sum(), P[y_ids].sum(), P[xy_map[x+y]].sum(), base=base)
        for x, x_ids in x_map.items() for y, y_ids in y_map.items()]) for P in system.Trajectory.trajectory])

    return mutual_information

def _summand(p_x, p_y, p_xy, base=2):
    if p_xy == 0:
        return 0
    return p_xy * math.log(p_xy / (p_x * p_y), base)

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

