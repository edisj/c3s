import pandas as pd
from c3s.simulators import MasterEquation
from c3s.calculations import calc_mutual_information_new
import argparse
from pathlib import Path


def benchmark():

    repeats = args.repeats
    filename = args.filename
    cfg = Path.cwd() / '../../data/AB_system.cfg'
    cfg_path = cfg.absolute()

    initial_t = 0
    final_t = 0.5
    dts = [1e-5, 1e-4, 1e-3, 1e-2]

    results = []
    columns = None

    for i in range(repeats):
        for dt in dts:
            initial_species = {'A':1, 'S':5, 'B':1}
            system = MasterEquation(initial_species=initial_species, cfg=cfg_path)
            system.run(start=initial_t, stop=final_t, step=dt)
            n_terms = len(system.results)
            X = ['A', 'A^*', 'AS', 'A^*S']
            Y = ['B', 'B^*']
            calc_mutual_information_new(system, X, Y)

            if columns is None:
                columns = list(system.timings.keys())
                columns = ['repeat', 'n_terms'] + columns

            times_for_this_iter = [i, n_terms] + list(system.timings.values())
            results.append(times_for_this_iter)

    df = pd.DataFrame(results, columns=columns)
    Path('serial/results').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'results/{filename}.csv', index=False)


if __name__ == "__main__":
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