#!/bin/bash
Robot=A1
gait=trot
num=01
mkdir motion_imitation/data/policies/$Robot/$gait$num
python3 motion_imitation/run.py --mode train --robot $Robot --motion_file motion_imitation/data/motions/$Robot/$gait.txt --output_dir motion_imitation/data/policies/$Robot/$gait$num --int_save_freq 10000000 --visualize