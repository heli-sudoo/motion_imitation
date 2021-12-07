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

def flip_sign_hip_knee(qJ_arr):
    """" Flip the signs of hip and knee if the trajectory data are obtained using A1"""
    qJ_arr_mc = np.copy(qJ_arr)
    qJ_arr_mc[:,[1,4,7,10]] = -qJ_arr[:,[1,4,7,10]]
    qJ_arr_mc[:,[2,5,8,11]] = -qJ_arr[:,[2,5,8,11]]

    return qJ_arr_mc

with open(rollout_fname, 'rb') as f:
    data = pkl.load(f)
    
time_arr, o_arr, a_arr, torque_arr, ctacts_arr, state_traj = data
rpy_arr, pos_arr, rpyrate_arr, vel_arr, qJ_arr, qJd_arr = state_traj
quat_arr = [pb.getQuaternionFromEuler(rpy) for rpy in list(rpy_arr)]
quat_arr = np.asarray(quat_arr)

qJ_arr_mc = flip_sign_hip_knee(qJ_arr)
pos_arr_mc = np.copy(pos_arr)
pos_arr_mc[:,0] = pos_arr_mc[:,0] + 0.0

replay_data_mc= ReplayData("mini_cheetah/mini_cheetah.urdf", pos_arr_mc, quat_arr, qJ_arr_mc)
replay_data_a1 = ReplayData("a1/a1.urdf", pos_arr, quat_arr, qJ_arr)
replay([replay_data_a1, replay_data_mc])  