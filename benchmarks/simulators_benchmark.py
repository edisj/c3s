import pandas as pd
import numpy as np
from c3s.simulators import MasterEquation
from c3s.calculations import calc_mutual_information
import argparse
from pathlib import Path
from mpi4py import MPI


def benchmark():

    repeats = args.repeats
    filename = args.filename
    cfg = Path.cwd() / '../c3s/data/AB_system.cfg'
    cfg_path = cfg.absolute()

    initial_t = 0
    final_t = 0.5
    dt = 0.001

    initial_species = {'A':1, 'S':30, 'B':1}
    system = MasterEquation(initial_species=initial_species, cfg=cfg_path)
    system.run(initial_time=initial_t, final_time=final_t, dt=dt)
    #n_states = len(system.constitutive_states)
    #n_terms = len(system.results)
    X = ['A', 'A^*', 'AS', 'A^*S']
    Y = ['B', 'B^*']
    calc_mutual_information(system, X, Y)
    #columns = ['rank', 'n_processes', 'n_states', 'n_terms'] + list(system.timings.keys())
    #times = np.array([rank, size, n_states, n_terms] + list(system.timings.values()))

    gathered_results = None
    #if rank == 0:
    #    n_rows = size
    #    n_cols = len(times)
        #gathered_results = np.empty(shape=(n_rows,n_cols), dtype=float)

    # Gather the results from each process into the data buffer
    #comm.Gather(sendbuf=times, recvbuf=gathered_results, root=0)
    '''if rank == 0:
        print('MADE IT HERE')
        df = pd.DataFrame(gathered_results, columns=columns)
        Path('results').mkdir(parents=True, exist_ok=True)
        df.to_csv(f'results/{filename}.csv', index=False)'''


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--filename", type=str, default=None)
    args = parser.parse_args()

    if args.repeats is None:
        raise ValueError("missing --repeats argument")
    if args.filename is None:
        raise ValueError("missing --filename argument")

    benchmark()
