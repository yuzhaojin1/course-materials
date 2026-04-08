#!/bin/bash
#SBATCH --job-name=random_walk_cupy
#SBATCH --output=random_walk_cupy_%j.out
#SBATCH --error=random_walk_cupy_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --account=macs30113

module load python/anaconda-2022.05 cuda/11.7
module load cuda/12.2

python cupy_gpu_pi.py --n 1000000
python cupy_gpu_pi.py --n 10000000