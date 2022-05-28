import pandas as pd
import numpy as np
from c3s.parallel_simulators import MasterEquationParallel
from c3s.calculations import calc_mutual_information_parallel
import argparse
from pathlib import Path
from mpi4py import MPI


def benchmark():

    repeats = args.repeats
    filename = args.filename
    cfg = Path.cwd() / '../../data/AB_system.cfg'
    cfg_path = cfg.absolute()

    initial_t = 0
    final_t = 0.5
    dt = 0.001

    substrate_quantities = [5, 10, 15, 20, 25, 30, 35, 40]
    results = []
    columns = None

    for repeat in range(repeats):
        for S in substrate_quantities:
            initial_species = {'A':1, 'S':S, 'B':1}
            system = MasterEquationParallel(initial_species=initial_species, cfg=cfg_path)
            system.run(start=initial_t, stop=final_t, step=dt)
            n_states = len(system.constitutive_states)
            X = ['A', 'A^*', 'AS', 'A^*S']
            Y = ['B', 'B^*']
            calc_mutual_information_parallel(system, X, Y)

            if columns is None:
                columns = list(system.timings.keys())
                columns = ['rank', 'n_processes', 'n_states'] + columns

            times = [rank, size, n_states] + list(system.timings.values())
            results.append(times)

    results = np.array(results, dtype=float)
    gathered_results = None
    if rank == 0:
        n_rows = len(substrate_quantities) * repeats * size
        n_cols = len(times)
        gathered_results = np.empty(shape=(n_rows,n_cols), dtype=float)

    # Gather the results from each process into the data buffer
    comm.Gather(sendbuf=results, recvbuf=gathered_results, root=0)

    if rank == 0:
        df = pd.DataFrame(gathered_results, columns=columns)
        Path('results').mkdir(parents=True, exist_ok=True)
        df.to_csv(f'results/{filename}.csv', index=False)


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
