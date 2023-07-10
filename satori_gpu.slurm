!/bin/bash
#SBATCH -J AS_4GPU # project name
#SBATCH -o AS_4GPU_%j.out # output file
#SBATCH -e AS_4GPU_%j.err # error file
#SBATCH --mail-user=jjzou@mit.edu # user email address
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBARCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# load modules
module load spack
module load cuda/10.1.243
# module load openmpi/3.1.4-pmi-cuda

# export
# export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}

julia -e 'using Pkg; pkg"instantiate"'
# julia -e 'using Pkg; pkg"build MPI"'
julia -e 'using Pkg; pkg"precompile"'

# Cleaning `CUDA_VISIBLE_DEVICES`
# This is needed to take advantage of faster local CUDA-aware communication
cat > launch.sh << EoF_s
#! /bin/sh
export CUDA_VISIBLE_DEVICES=0
exec \$*
EoF_s
chmod +x launch.sh

EXPERIMENT="${HOME2}/ActiveSubspaces.jl/examples/doublewell1D/01_compute_as.jl --output-dir=${HOME2}/as-${SLURM_JOB_ID}"

julia ${EXPERIMENT}

