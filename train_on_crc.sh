#!/bin/bash

#$ -M hli25@nd.edu
#$ -m abe
#$ -pe mpi-24 48
#$ -q long
#$ -t 1-2            # Specify number of tasks in array
#$ -N mc_trot 	       # job name
module load mpich/3.3/intel/19.0
conda activate RLMPC
export OMP_NUM_THREADS=${NSLOTS}

Gait=trot
Robot=MC
mpiexec -n 48 python3 motion_imitation/run.py --mode train --robot $Robot --motion_file motion_imitation/data/motions/$Robot/$Gait.txt --output_dir motion_imitation/data/policies/intermediate/$Robot/$Gait_$SGE_TASK_ID --int_save_freq 10000000
