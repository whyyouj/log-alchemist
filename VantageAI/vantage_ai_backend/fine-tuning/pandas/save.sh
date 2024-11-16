#!/bin/sh
#SBATCH --time=40:00:00
#SBATCH --job-name=mytestjob
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@comp.nus.edu.sg
#SBATCH --gpus=1

python train.py

