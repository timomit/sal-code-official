#!/bin/bash
#SBATCH --job-name="symmnet sweep"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --account=paygo
#SBATCH --wckey=pyl_biodl

# Your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"

conda activate ../../envs/

echo "BLA!"

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")
eval "$CMD"
