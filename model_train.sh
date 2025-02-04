#!/bin/bash

#$ -M hli25@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -q long
#$ -N motion_imitation_parallel
module load mpich/3.3/intel/19.0
conda activate RLMPC
export OMP_NUM_THREADS=${NSLOTS}

mpiexec -n 8 python3 motion_imitation/run.py --mode train --motion_file motion_imitation/data/motions/dog_pace.txt --int_save_freq 10000000
