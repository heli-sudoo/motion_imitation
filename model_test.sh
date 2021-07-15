#!/bin/bash

module load mpich/3.3/intel/19.0
source /opt/crc/c/conda/miniconda3/4.9.2/etc/profile.d/conda.sh
conda activate RLMPC
export MESA_GL_VERSION_OVERRIDE=3.3

python3 motion_imitation/run.py --mode test --motion_file motion_imitation/data/motions/dog_pace.txt --model_file motion_imitation/data/policies/dog_pace.zip --visualize
