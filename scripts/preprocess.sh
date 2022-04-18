#!/bin/bash
# A shell script to run the python preprocessing script via slurm

#SBATCH --job-name=prepro
#SBATCH --partition=LocalQ
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/preprocess_%j.out
#SBATCH --open-mode=append


ml python/3.6
date
python3 -u ./preprocess.py $@
