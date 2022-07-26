#!/bin/bash
# A shell script to run the python preprocessing script via slurm

#SBATCH --job-name=prepro
#SBATCH --partition=trc
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=sbatch_preproc_%j.log   # Standard output and error log


# ml python/3.6
date
echo "running preprocess.py"
python3 -u ./preprocess.py $@
