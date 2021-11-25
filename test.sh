#!/bin/bash
Mode=rollout
Robot=A1
Gait=pace
Policy=pace_04
python3 motion_imitation/run.py --mode $Mode --robot $Robot --motion_file motion_imitation/data/motions/$Robot/$Gait.txt --model_file motion_imitation/data/policies/$Robot/$Policy.zip --visualize