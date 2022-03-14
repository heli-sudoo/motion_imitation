#!/bin/bash
if [ $# -eq 0 ]; then
   echo "Usage: test.sh mode robot gait policy"
   echo "options:"
   echo "         mode: test rollout"
   echo "         robot: A1 MC Laikago"
   echo "         gait: trot pace"
   echo "         policy: policy fille under motion_imitation/data/policies"
else
   Mode=$1
   Robot=$2
   Gait=$3
   Policy=$4
   python3 motion_imitation/run.py --mode $Mode --robot $Robot --motion_file motion_imitation/data/motions/$Robot/$Gait.txt --model_file motion_imitation/data/policies/$Policy.zip --visualize
fi

