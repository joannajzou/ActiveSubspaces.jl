!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mem=1T
#SBATCH --mail-user=jjzou@mit.edu # user email address
#SBATCH --mail-type=ALL
#SBATCH -J lmps_ar # project name
#SBATCH -o lmps_ar_%j.out # output file
#SBATCH -e lmps_ar_%j.err # error file
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH -c 12


# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# load modules
module load spack
module load openmpi

# run scripts
for i in {1..3};
do  
	mpirun -n 12 lmp -sf gpu -pk gpu 2 -in <in_.script>


