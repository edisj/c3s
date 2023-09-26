from typing import  Dict, Tuple
import numpy as np


def get_point_mappings(species_subset, system) -> Dict[Tuple, np.ndarray]:
    """Maps each microstate vector in `system.states` to their marginal probability distribution
    domain point for the specified molecular species if you were to sum over all other species != `species`.

    e.g. find where each (c_A, c_B, c_C, c_D) point in p(c_A, c_B, c_C, c_D) maps to in p(c_A, c_B) if the set
    of (c_A, c_B, c_C, c_D) points were flattened into a rank-1 array where c_C and c_D are summed over.

    """
    if system._Trajectory is None:
        raise AttributeError("no data in `system.trajectory`")
    species_subset = [species_subset] if isinstance(species_subset, str) else species_subset
    ids = [system.species.index(species) for species in list(species_subset)]
    subspace = np.unique(system.states[:, ids], axis=0)
    point_mappings = {
        tuple(vector) : np.where(np.all(system.states[:, ids] == vector, axis=1))[0] for vector in subspace}
    return point_mappings


def get_marginalized_trajectory(species_subset, system) -> Dict[Tuple, np.ndarray]:
    point_mappings = get_point_mappings(species_subset, system)
    marginalized_trajectory = {state: np.asarray([P_t[ids].sum() for P_t in system.trajectory])
                               for state, ids in point_mappings.items()}
    return marginalized_trajectory


def get_average_copy_number(species, system) -> np.ndarray:
    """calculates the average copy number for a single chemical species over the probability trajectory"""
    marginalized_trajectory = get_marginalized_trajectory(species, system)
    avg_copy_number = np.asarray([sum([c[0]*P_c[ts] for c, P_c in marginalized_trajectory.items()])
                                  for ts in range(len(system.trajectory))])
    return avg_copy_number
