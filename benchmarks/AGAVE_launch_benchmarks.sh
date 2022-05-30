#!/bin/bash

cd /scratch/ejakupov/c3s/benchmarks

for n_cpus in 1 4 8 12 16 20 24 28
do
  for repeat in 0 1 2
  do
    sbatch -N 1 -n $n_cpus AGAVE_run_benchmark.sh $repeat
  done
done

for n_cpus in 32 36 40 44 48
do
  for repeat in 0 1 2
  do
    sbatch -N 2 -n $n_cpus AGAVE_run_benchmark.sh $repeat
  done
done

for n_cpus in 88 92 96 100 104 108 112
do
  for repeat in 0 1 2
  do
    sbatch -N 4 -n $n_cpus AGAVE_run_benchmark.sh $repeat
  done
done