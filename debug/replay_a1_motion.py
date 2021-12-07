from replay_motion import ReplayData 
from replay_motion import replay

import pickle as pkl
import os
import inspect

import pybullet as pb
import numpy as np

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
rollout_fname = parentdir + '/motion_imitation/data/rollout/traj_data.pickle'

with open(rollout_fname, 'rb') as f:
    data = pkl.load(f)
    
time_arr, o_arr, a_arr, torque_arr, ctacts_arr, state_traj = data
rpy_arr, pos_arr, rpyrate_arr, vel_arr, qJ_arr, qJd_arr = state_traj
quat_arr = [pb.getQuaternionFromEuler(rpy) for rpy in list(rpy_arr)]
quat_arr = np.asarray(quat_arr)
  
replay_data = ReplayData("a1/a1.urdf", pos_arr, quat_arr, qJ_arr)
replay([replay_data])  