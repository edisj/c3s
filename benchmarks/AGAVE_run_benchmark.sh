#!/bin/bash
#SBATCH -t 0-4:00:00         # time in d-hh:mm:ss
#SBATCH -p parallel          # partition
#SBATCH -q normal            # QOS
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ejakupov@asu.edu
#SBATCH --output=/scratch/ejakupov/c3s/benchmarks/results/logs/%j.log
#SBATCH --time=2:00:00

# useful function for quitting with an error message
die () {
    local msg="$1" err=${2:-1}
    echo "ERROR [$err]: $msg"
    exit $err
}

cd /scratch/ejakupov/c3s/benchmarks
module load
module load
source activate c3s

NODES=$SLURM_NNODES
CORES=$SLURM_NPROCS
REPEAT=$1

# check that they are not empty and exist
test -n "$REPEAT" || die "No repeat number supplied"

echo "---------------"
echo "REPEAT=$REPEAT"
echo "NODES=$NODES"
echo "CORES=$CORES"
JOB_NAME="${NODES}nodes_${CORES}cores_${REPEAT}"
echo "JOB_NAME=${JOB_NAME}"
echo "---------------"

mpiexec -n $CORES python benchmark.py --filename $JOB_NAME
