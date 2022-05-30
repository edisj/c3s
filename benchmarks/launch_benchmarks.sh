#!/usr/bin/bash

n_cpus=$1

for repeat in 0 1 2
do
  . run_benchmark.sh $n_cpus $repeat
done