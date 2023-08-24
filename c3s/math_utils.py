import numpy as np


def binary_search(vector, x, low=0, high=None):
    """"""
    if high is None:
        high = len(vector)-1
    if high >= low:
        mid = (high + low) // 2
        if vector[mid] == x:
            return mid
        elif vector[mid] > x:
            return binary_search(vector, x, low=low, high=mid-1)
        else:
            return binary_search(vector, x, low=mid+1, high=high)
    else:
        return False


def cartesian_product(space_1, space_2):
    product_space = [np.concatenate((state_i, state_j)) for state_i in space_1 for state_j in space_2]
    return np.stack(product_space)


def combine_state_spaces(*subspaces):
    if len(subspaces) == 2:
        return cartesian_product(subspaces[0], subspaces[1])
    return cartesian_product(subspaces[0], combine_state_spaces(*subspaces[1:]))


def vector_to_number(vector, N, base):
    number = (vector * (base**np.arange(N-1, -1, -1))).sum(axis=1)
    return number
