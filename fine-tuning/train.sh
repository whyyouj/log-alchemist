#!/bin/sh
#SBATCH --time=20:00:00
#SBATCH --job-name=mytestjob
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@comp.nus.edu.sg
#SBATCH --gpus=1

#which python
#source /home/j/jiayuan1/.jy/jy_venv/bin/activate

#python hello_world.py
python train.py
#python data.py
