import c3s
import numpy as np
from mpi4py import MPI
import h5py


def parameter_sweep(S):

    base_rate = 25
    rates = base_rate * np.array([0.01, 0.1, 1, 10, 100])

    X = ['A', 'A*', 'A*S', 'AS']
    Y = ['B', 'BP']

    system = c3s.simulators.MasterEquation(cfg='testing/config_files/enzyme_with_allostery.yml', A=1, B=1, S=S)

    with h5py.File(f'parameter_sweep_{S}substrate.h5', mode='x') as f:
        pass
