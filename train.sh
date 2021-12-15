#!/bin/bash
if [ $# -eq 0 ]; then
   echo "Usage: train.sh robot gait policy_dir"
   echo "options:"
   echo "         robot: A1 MC Laikago"
   echo "         gait: trot pace"
   echo "         policy_dir: directory where the policy would be saved to"
   echo "example: train MC trot trot01"
else
   Robot=$1
   Gait=$2
   Policy_dir=$3
fi
mkdir motion_imitation/data/policies/$Robot/$Policy_dir
python3 motion_imitation/run.py --mode train --robot $Robot --motion_file motion_imitation/data/motions/$Robot/$Gait.txt --output_dir motion_imitation/data/policies/$Robot/$Policy_dir --int_save_freq 10000000 --visualize