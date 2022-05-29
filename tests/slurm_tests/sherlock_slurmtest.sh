#!/bin/bash
# A shell script to run the python preprocessing script via slurm

#SBATCH --job-name=longtest
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=longtest_%j.log   # Standard output and error log


# ml python/3.6
date
echo "running sherlock_slurmtest.py"
python3 -u ./sherlock_slurmtest.py
