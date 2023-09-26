from collections import namedtuple
from typing import List, Dict, Tuple

import numpy as np
from .math_utils import generate_subspace, combine_state_spaces
from .simulators.reaction_network import ReactionNetwork


def get_PointMappings(system, X, Y=None):
    """Maps each microstate vector in `system.states` to their marginal probability distribution
    domain point for the specified molecular species if you were to sum over all other species != `species`.

    e.g. find where each (c_1,c_2,c_3) point in p(c_1, c_2, c_3) maps to in p(c_1,c_2) if the set
    of (c_1,c_2,c_3) points were flattened into a rank-1 array where c_3 is marginalized/summed over.

    Parameters
    ----------
    system : :class:`c3s.ChemicalMasterEquation`
        instance of ChemicalMasterEquation
    X : List[str]
        subset of `system.species`
    Y : List[str] (default=None)
        subset of `system.species`, different from X

    Returns
    -------
    PointMappings : NamedTuple
        always contains PointMappings.X, contains PointMappings.Y and PointMappings.XY if Y is not None

    """

    PointMappings = namedtuple('PointMappings', ['X', 'Y', 'XY'])

    # convert string to list just in case argument given as string e.g. 'S' -> ['S']
    X = [X] if isinstance(X, str) else X
    point_mappings_X = _map_subspace_to_indices(
        system=system,
        species_subset=X,
        subspace=generate_subspace(_Constraint(system, X)))
    if Y is not None:
        Y = [Y] if isinstance(Y, str) else Y
        point_mappings_Y = _map_subspace_to_indices(
            system=system,
            species_subset=Y,
            subspace=generate_subspace(_Constraint(system, Y)))
        point_mappings_XY = _map_subspace_to_indices(
            system=system,
            species_subset=X+Y,
            subspace=combine_state_spaces(
                generate_subspace(_Constraint(system, X)), generate_subspace(_Constraint(system, Y))))
    else:
        point_mappings_Y = None
        point_mappings_XY = None

    return PointMappings(X=point_mappings_X, Y=point_mappings_Y, XY=point_mappings_XY)


def _map_subspace_to_indices(system, species_subset, subspace):
    point_mappings: Dict[Tuple: np.ndarray] = {}
    # species_subset_ids gives the columns of `system.states` that correspond to species_subset
    species_subset_ids = [system.species.index(species) for species in species_subset]
    for subset_vector in subspace:
        # idea here is to find all states in system.states where the copy numbers
        # are equal to the valid  copy numbers in the subset state space
        # e.g. find indices of all states where c_A=2 for a single species
        # or (c_A=2) & (c_B=1) for two species
        ids_where_copy_numbers_equal = np.where(np.all(system.states[:, species_subset_ids] == subset_vector, axis=1))
        # np.where above returns tuple(array) for some reason, so I remove the tuple by slicing [0]
        point_mappings[tuple(subset_vector)] = ids_where_copy_numbers_equal[0]

    return point_mappings


def _Constraint(system, species_subset):
    for _constraint in system.reaction_network.constraints:
        if all([species in _constraint.species_involved for species in species_subset]):
            separator = '<=' if len(species_subset) == 1 else _constraint.separator
            constraint = f"{' + '.join(species_subset)} {separator} {_constraint.value}"
            species_involved = species_subset
            value = _constraint.value
            Constraint = ReactionNetwork.Constraint(
                 constraint=constraint, species_involved=species_involved, separator=separator, value=value)
            # acting as for-loop break statement
            return Constraint
    else:
        raise ValueError(f"valid constraint not found for `species_subset={species_subset}`")


def marginalize_trajectory(system, species_subset):
    """
    Parameters
    ----------
    system : :class:`c3s.ChemicalMasterEquation`
        instance of ChemicalMasterEquation
    species_subset : List[str]
        subset of `system.species`

    Returns
    -------
    marginalized_trajectory : Dict[tuple: np.ndarray]
        `system.Trajectory.trajectory` marginalized over all species != `species_subset` where
        `marginalized_trajectory.keys()` are the feasible subset vectors and
        `margainlized_trajectory.values()` are the probability trajectories of those subset vectors

    """
    marginalized_trajectory: Dict[tuple: np.ndarray] = {}

    PointMappings = get_PointMappings(system=system, X=species_subset)
    for state, ids in PointMappings.X.items():
        marginalized_trajectory[state] = np.array([P_t[ids].sum() for P_t in system.Trajectory.trajectory])

    return marginalized_trajectory


def average_copy_number(system, species):
    """calculates the average copy number for a single chemical species over the probability trajectory

    Parameters
    ----------
    system : :class:`c3s.ChemicalMasterEquation`
        instance of ChemicalMasterEquation
    species : List[str]
        subset of `system.species`

    Returns
    -------
    average_copy_number : np.ndarray
        timeseries of the average copy number of species `species` over the trajectory

    """
    marginalized_trajectory = marginalize_trajectory(system=system, species_subset=species)
    for state in marginalized_trajectory.keys():
        # quickly make sure we're dealing with a single species
        assert(len(state) == 1)
    N_timesteps = len(system.Trajectory.trajectory)
    acg_copy_number = np.empty(shape=N_timesteps, dtype=np.float64)

    # probability a faster way to do this using vectorization/broadcasting
    for ts in range(N_timesteps):
        total = 0
        for state, P in marginalized_trajectory.items():
            copy_number = state[0]
            total += np.sum(copy_number * P[ts])
        acg_copy_number[ts] = total

    return acg_copy_number
