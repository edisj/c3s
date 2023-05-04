import numpy as np
from collections import namedtuple
from typing import List, Dict, NamedTuple


def get_Delta_vectors(X: List[str], Y: List[str], system):
    """generates full set of Delta vectors for X, Y and their joint X+Y

    Args:
        X, Y:
            two arbitrary subsets of chemical species
    Returns:
        Deltas:
            namedtuple with attributes Deltas.X, Deltas.Y, Deltas.XY which each correspond
            to a dictionary that maps marginalized coordinates to Delta vectors

    """

    X, Y = _check_species_sorting(X, Y)

    species = system.species
    states = system.states
    DeltaTuple = namedtuple("Deltas", "X Y XY")
    Deltas = DeltaTuple(
        _map_marginalized_coordinates_to_Delta_vectors(X, system),
        _map_marginalized_coordinates_to_Delta_vectors(Y, system),
        _map_marginalized_coordinates_to_Delta_vectors(X+Y, system))

    return Deltas


def _map_marginalized_coordinates_to_Delta_vectors(X, system):
    # maybe need docstring here

    species = system.species
    states = system.states
    tuple_to_Deltas_map: Dict[tuple: np.ndarray] = {}

    # indices that correspond to the selected molecular species in the ordered species list
    indices_we_care_about = [species.index(n) for n in X]
    # only consider elements that corresepond to species in X
    truncated_state_space = states[:, indices_we_care_about]

    curr_state_id = 0
    states_accounted_for = []
    # keep looping until every state in the microstate space is accounted for
    while len(states_accounted_for) < len(truncated_state_space):
        # find the indices of degenerate domain points starting with the first state
        # and skipping iterations if that point has been accounted for
        if curr_state_id in states_accounted_for:
            curr_state_id += 1
            continue
        curr_state = truncated_state_space[curr_state_id]
        # Delta is selecting for states that marginalize over the species we are ignoring
        Delta = np.all(truncated_state_space == curr_state, axis=1).astype(np.float64)
        tuple_to_Deltas_map[tuple(curr_state)] = Delta
        states_accounted_for += np.argwhere(Delta == 1).transpose().tolist()
        curr_state_id += 1

    return tuple_to_Deltas_map


def _check_species_sorting(X: List[str], Y: List[str]) -> (List[str], List[str]):
    """important helper function that sorts species arguments

    chemical species within the lists X and Y must be sorted in the same order as when
    the simulation was run so that the indices used in calculations correspond to the correct species

    Args:
        X, Y:
            two arbitrary subsets of chemical species

    Returns:
        X, Y:
            the same lists of chemical species, but their elements may have been rearranged

    """
    if isinstance(X, str):
        X = [X]
    if isinstance(Y, str):
        Y = [Y]
    X = sorted(X)
    Y = sorted(Y)
    if X + Y != sorted(X + Y):
        X, Y = Y, X

    return X, Y


def average_population(species: str, system) -> np.ndarray:
    """calculates the average population number for a particular chemical species

    Args:
        species:
            a string that corresponds to a chemical species from the simulation
    Returns:
        average_population:
            the average population of species `species` over the trajectory

    """

    assert isinstance(species, str)
    species = [species]
    trajectory = system.trajectory
    Delta_vectors = _map_marginalized_coordinates_to_Delta_vectors(species, system)
    average_population = np.empty(shape=len(trajectory), dtype=np.float64)

    for i, P in enumerate(trajectory):
        total = 0
        # iterate through every possible population number
        for coordinate, Delta in Delta_vectors.items():
            assert len(coordinate) == 1
            count = coordinate[0]
            if count != 0:
                this_term = count * np.dot(P, Delta)
                total += this_term

        average_population[i] = total

    return average_population

def _average_populationOLD(species: str, system) -> np.ndarray:

    trajectory = system.trajectory
    average_population = np.empty(shape=len(trajectory), dtype=np.float64)
    if trajectory is None:
        raise ValueError('No data found in self.trajectory.')

    point_maps = _get_point_mappings(species, system)
    for ts in range(len(trajectory)):
        total = 0
        for point, map in point_maps.items():
            assert len(point) == 1
            count = point[0]
            count_term = np.sum(count * trajectory[ts][map])
            total += count_term
        average_population[ts] = total

    return average_population

def calculate_marginal_probability_evolution(X, system):

    trajectory = system.trajectory
    if trajectory is None:
        raise ValueError('No data found in self.trajectory.')
    point_maps = _get_point_mappings(X, system)
    distribution: Dict[tuple, np.ndarray] = {}
    for point, map in point_maps.items():
        distribution[point] = np.array([np.sum(trajectory[ts][map]) for ts in range(len(trajectory))])

    return distribution


def _get_point_mappings(X, system):
    """

    Maps the indices of the microstates in the `system.states` vector to their
    marginal probability distribution domain points for the specified molecular species.

    e.g. find where each (x,y,z) point in p(x,y,z) maps in p(x) if the set of (x,y,z) points were
    flattened into a 1-d array.

    """

    if isinstance(X, str):
        molecules = [X]

    species = system.species
    states = system.states
    # indices that correspond to the selected molecular species in the ordered species list
    ids = [species.index(n) for n in sorted(X)]
    truncated_points = states[:, ids]

    # keys are tuple coordinates of the marginal distrubtion
    # values are the indices of the joint distribution points that map to the marginal point
    point_maps: Dict[tuple, List[int]] = {}

    curr_state_id = 0
    states_accounted_for: List[int] = []
    # begin mapping points until every state in the microstate space is accounted for
    while len(states_accounted_for) < len(truncated_points):
        # find the indices of degenerate domain points starting with the first state
        # and skipping iterations if that point has been accounted for
        if curr_state_id in states_accounted_for:
            curr_state_id += 1
            continue
        curr_state = truncated_points[curr_state_id]
        degenerate_ids = np.argwhere(
            np.all(truncated_points == curr_state, axis=1)).transpose().flatten().tolist()
        point_maps[tuple(curr_state)] = degenerate_ids
        # keep track of which states we have accounted for
        states_accounted_for += degenerate_ids
        curr_state_id += 1

    return point_maps
