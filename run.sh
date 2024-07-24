#!/bin/bash -l
#SBATCH --job-name=Rec21_200k
#SBATCH --time=72:00:00
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH -A mkamino1_gpu

module load anaconda/2020.07
module load cuda/12.1.0
module load cudnn/8.0.4.30-11.1-linux-x64
conda activate ML
time python run.py
