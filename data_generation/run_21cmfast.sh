#!/bin/bash -l
#SBATCH --job-name=21cmfast
#SBATCH --time=00-72:00:00
#SBATCH --partition=parallel ##shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --account=mkamion1
#SBATCH --array=1-5000
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load anaconda/2020.07
conda activate 21cmfast
time python run.py
