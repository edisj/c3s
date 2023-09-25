import numpy as np
from numba import njit, typed, types
from numba.experimental import jitclass


"""citation  https://github.com/PessoaP/smn 
TODO: FILL IN

"""

@njit
def sm_asarray(rows, columns, values, shape):
    res = np.zeros((shape, shape))
    for i in range(rows.size):
        res[rows[i], columns[i]] += values[i]
    return res


@njit
def sm_sum(rows, columns, values, shape):
    res = np.zeros(shape)
    ind = rows
    for i in range(rows.size):
        res[ind[i]] += values[i]
    return res


@njit
def multiply_sparse_matrices(rows1, columns1, values1, rows2, columns2, values2):
    res_rows = []
    res_cols = []
    res_vals = []

    for row1, col1, val1 in zip(rows1, columns1, values1):
        for row2, col2, val2 in zip(rows2, columns2, values2):
            if col1 == row2:
                res_rows.append(row1)
                res_cols.append(col2)
                res_vals.append(val1 * val2)

    return np.array(res_rows), np.array(res_cols), np.array(res_vals)


@njit
def simplify_sparse_matrix(rows, columns, values):
    seen_indices = {}  # dictionary to be filled at the first apparence of a row and columns pairing

    for row, col, val in zip(rows, columns, values):
        if (row, col) in seen_indices:
            seen_indices[(row, col)] += val
        else:
            seen_indices[(row, col)] = val

    simplified_rows = []
    simplified_cols = []
    simplified_values = []

    for (r, c), v in zip(seen_indices.keys(), seen_indices.values()):
        if v != 0:
            simplified_rows.append(r)
            simplified_cols.append(c)
            simplified_values.append(v)
    return np.array(simplified_rows), np.array(simplified_cols), np.array(simplified_values)


@jitclass([('rows', types.int64[:]),
           ('columns', types.int64[:]),
           ('values', types.float64[:]),
           ('shape', types.int64)])
class SparseMatrix:
    def __init__(self, rows, cols, values, shap=0):
        self.rows = rows
        self.columns = cols
        self.values = values
        self.shape = shap
        if shap == 0:
            self.shape = 1 + rows.max()  # assume always square

    def __add__(self, other):
        return SparseMatrix(np.concatenate((self.rows, other.rows)),
                             np.concatenate((self.columns, other.columns)),
                             np.concatenate((self.values, other.values)),
                             max(self.shape, other.shape))

    def __sub__(self, other):
        return SparseMatrix(np.concatenate((self.rows, other.rows)),
                             np.concatenate((self.columns, other.columns)),
                             np.concatenate((self.values, -other.values)),
                             max(self.shape, other.shape))

    def to_dense(self):
        return sm_asarray(self.rows, self.columns, self.values, self.shape)

    def row_sum(self):
        return sm_sum(self.rows, self.columns, self.values, self.shape)

    def column_sum(self):
        return sm_sum(self.columns, self.rows, self.values, self.shape)

    def simplify(self):
        self.rows, self.columns, self.values = simplify_sparse_matrix(self.rows, self.columns, self.values)
        ind1 = np.argsort(self.columns)
        self.rows, self.columns, self.values = self.rows[ind1], self.columns[ind1], self.values[ind1]
        ind2 = np.argsort(self.rows)
        self.rows, self.columns, self.values = self.rows[ind2], self.columns[ind2], self.values[ind2]

    def __mul__(self, other):
        if other.shape != self.shape:
            print('Warning: you are trying to multiply square matrices of different shapes')
        prod = SparseMatrix(
            *multiply_sparse_matrices(self.rows, self.columns, self.values, other.rows, other.columns, other.values),
            self.shape)
        prod.simplify()
        return prod


@njit
def array_times_sm(arr, sm):
    res = np.zeros(sm.shape, types.float64)
    for (row, col, val) in zip(sm.rows, sm.columns, sm.values):
        res[col] += val * arr[row]
    return res


@njit
def sm_times_array(sm, arr):
    res = np.zeros(sm.shape, types.float64)
    for (row, col, val) in zip(sm.rows, sm.columns, sm.values):
        res[row] += val * arr[col]
    return res


def sum(sm, axis=-1):
    if axis == 0:
        return sm.column_sum()
    elif axis == 1:
        return sm.row_sum()
    else:
        return sm.row_sum().sum()


def dot(A, B):
    if isinstance(A, SparseMatrix):
        if isinstance(B, SparseMatrix):
            return A * B
        if isinstance(B, np.ndarray):
            return sm_times_array(A, B)
    elif isinstance(A, np.ndarray):
        if isinstance(B, SparseMatrix):
            return array_times_sm(A, B)
        if isinstance(B, np.ndarray):
            return np.dot(A, B)
