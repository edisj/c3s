import pandas as pd
import numpy as np
from c3s import MasterEquation
from c3s import calc_mutual_information
import argparse
from pathlib import Path
from mpi4py import MPI


def benchmark():

    filename = args.filename
    cfg = Path.cwd() / '../c3s/data/AB_system.cfg'
    cfg_path = cfg.absolute()

    initial_t = 0
    final_t = 0.5
    #dts = [0.01, 0.001, 0.0001, 0.00001]
    #substrate_quantities = [5, 10, 15, 20]
    dts = [0.0001]
    substrate_quantities = [20]

    results = []
    columns = None
    for S in substrate_quantities:
        for dt in dts:
            initial_species = {'A':1, 'S':S, 'B':1}
            system = MasterEquation(initial_species=initial_species, cfg=cfg_path)
            system.run(initial_time=initial_t, final_time=final_t, dt=dt)
            n_states = len(system.constitutive_states)
            n_terms = len(system.results)
            X = ['A', 'A^*', 'AS', 'A^*S']
            Y = ['B', 'B^*']
            calc_mutual_information(system, X, Y)
            if columns is None:
                columns = ['rank', 'n_processes', 'n_states', 'n_terms'] + list(system.timings.keys())
            times = [rank, size, n_states, n_terms] + list(system.timings.values())
            results.append(times)

    results = np.array(results)
    gathered_results = None
    if rank == 0:
        n_rows =  size * len(dts) * len(substrate_quantities)
        n_cols = len(columns)
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
    parser.add_argument("--filename", type=str, default=None)
    args = parser.parse_args()

    if args.filename is None:
        raise ValueError("missing --filename argument")

    benchmark()
