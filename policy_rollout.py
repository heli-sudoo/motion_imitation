#!/usr/bin/python3
import numpy as np

import pickle as pkl
import matplotlib.pyplot as plt

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
rollout_fname = currentdir + '/motion_imitation/data/rollout/traj_data.pickle'

LEG_INDEX = {'FR': 0, 'FL': 1, 'RR': 2, 'RL': 3}


def plot_joint_torques(time, torque):
    """ Plot joint torques versus time
    time: time step in Nx1 numpy array 
    torque: joint torque in Nxn numpy array 
    """
    fig, axs = plt.subplots(2, 2)
    for legname, legid in LEG_INDEX.items():
        axs[int(legid/2), legid % 2].plot(time, torque[:, 3*legid:3*legid+3])
        axs[int(legid/2), legid % 2].set_xlabel('time (s)')
        axs[int(legid/2), legid % 2].set_ylabel('torque (Nm)')
        axs[int(legid/2), legid % 2].set_title(legname)
        axs[0, 0].legend(['abad', 'hip', 'knee'])


def plot_joint_angles(time, q):
    """ Plot joint torques versus time
    time: time step in Nx1 numpy array 
    q: joint angles in Nxn numpy array 
    """
    fig, axs = plt.subplots(2, 2)
    for legname, legid in LEG_INDEX.items():
        axs[int(legid/2), legid % 2].plot(time, q[:, 3*legid:3*legid+3])
        axs[int(legid/2), legid % 2].set_xlabel('time (s)')
        axs[int(legid/2), legid % 2].set_ylabel('q (rad)')
        axs[int(legid/2), legid % 2].set_title(legname)
        axs[0, 0].legend(['abad', 'hip', 'knee'])


def plot_joint_vels(time, qd):
    """ Plot joint torques versus time
    time: time step in Nx1 numpy array 
    torque: joint torque in Nxn numpy array 
    """
    fig, axs = plt.subplots(2, 2)
    for legname, legid in LEG_INDEX.items():
        axs[int(legid/2), legid % 2].plot(time, qd[:, 3*legid:3*legid+3])
        axs[int(legid/2), legid % 2].set_xlabel('time (s)')
        axs[int(legid/2), legid % 2].set_ylabel('qd (rad/s)')
        axs[int(legid/2), legid % 2].set_title(legname)
        axs[0, 0].legend(['abad', 'hip', 'knee'])


def plot_body_pos(time, pos):
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(time, pos[:, 0])
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('x (m)')

    axs[1].plot(time, pos[:, 1])
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('y (m)')

    axs[2].plot(time, pos[:, 2])
    axs[2].set_xlabel('time (s)')
    axs[2].set_ylabel('z (m)')


def plot_body_rpy(time, rpy):
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(time, rpy[:, 0])
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('roll (rad)')

    axs[1].plot(time, rpy[:, 1])
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('pitch (rad)')

    axs[2].plot(time, rpy[:, 2])
    axs[2].set_xlabel('time (s)')
    axs[2].set_ylabel('yaw (rad)')


def plot_body_vel(time, vel):
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(time, vel[:, 0])
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('vx (m/s)')

    axs[1].plot(time, vel[:, 1])
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('vy (m/s)')

    axs[2].plot(time, vel[:, 2])
    axs[2].set_xlabel('time (s)')
    axs[2].set_ylabel('vz (m/s)')


def plot_body_rpy_rate(time, rpyrate):
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(time, rpyrate[:, 0])
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('rollrate (rad/s)')

    axs[1].plot(time, rpyrate[:, 1])
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('pitchrate (rad/s)')

    axs[2].plot(time, rpyrate[:, 2])
    axs[2].set_xlabel('time (s)')
    axs[2].set_ylabel('yawrate (rad/s)')

def plot_contact_sequences(time, ctacts):
    fig, ax = plt.subplots()
    for legname, legidx in LEG_INDEX.items():
        c = (np.reshape(ctacts[:,legidx], (1,-1))).astype(int)
        c = np.repeat(c,2,axis=0) 
        ax.pcolormesh(time, np.array([legidx-0.25, legidx+0.25]), c, shading='auto')
        ax.set_xlabel('time (s)')
        ax.set_ylim([-1, 4])
        ax.set_yticks([-1, 0, 1, 2, 3, 4])
        ax.set_yticklabels(('','FR','FL','RR','RL',''))



with open(rollout_fname, 'rb') as f:
    data = pkl.load(f)
time_arr, o_arr, a_arr, torque_arr, ctacts_arr, state_traj = data
rpy_arr, pos_arr, rpyrate_arr, vel_arr, q_arr, qd_arr = state_traj

# # plot trajectories
# plot_joint_torques(time_arr, torque_arr)
# plot_joint_angles(time_arr, q_arr)
# plot_joint_vels(time_arr, qd_arr)
# plot_body_pos(time_arr, pos_arr)
# plot_body_rpy(time_arr, rpy_arr)
# plot_body_vel(time_arr, vel_arr)
# plot_body_rpy_rate(time_arr, rpyrate_arr)
# plot_contact_sequences(time_arr, ctacts_arr)
# plt.tight_layout()
# plt.show()


# save roll-out trajectory to txt file
rollout_dir = '/home/wensinglab/HL/Code/HSDDP/MATLAB/Examples/Quadruped/Pacing/RolloutTrajectory'
torque_fname = rollout_dir + 'torque.txt'
ctact_fname = rollout_dir + 'contact.txt'
gjoint_fname = rollout_dir + 'generalized_joint.txt'
gvel_fname = rollout_dir + 'generalized_vel.txt'
time_fname = rollout_dir + 'timestep.txt'

with open(time_fname, 'w') as ftime:
    np.savetxt(ftime, time_arr, fmt = '%10.6f')

with open(torque_fname, 'w') as ft:
    np.savetxt(ft, torque_arr, fmt = '%10.6f')

with open(ctact_fname, 'w') as fc:
    np.savetxt(fc, ctacts_arr, fmt = '%d')

with open(gjoint_fname, 'w') as fj:
    np.savetxt(fj, np.hstack((pos_arr, rpy_arr, q_arr)), fmt = '%10.6f')

with open(gvel_fname, 'w') as fv:
    np.savetxt(fv, np.hstack((vel_arr, rpyrate_arr, qd_arr)), fmt = '%10.6f')