#!/usr/bin/bash

# useful function for quitting with an error message
die () {
    local msg="$1" err=${2:-1}
    echo "ERROR [$err]: $msg"
    exit $err
}

CORES=$1
REPEAT=$2

# check that they are not empty and exist
test -n "$CORES" || die "No cores supplied"
test -n "$REPEAT" || die "No repeat number supplied"

echo "--------------------"
echo "REPEAT=$REPEAT"
echo "CORES=$CORES"

JOB_NAME="${CORES}cores_${REPEAT}"
echo "JOB_NAME=${JOB_NAME}"

cd ~/Projects/c3s/benchmarks/
mpiexec -n $CORES python benchmark.py --filename $JOB_NAME

unset CORES
unset REPEAT
