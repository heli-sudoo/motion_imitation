#!/bin/usr/python3

""" Joint PD control on simulated mini cheetah in Pybullet """

from math import tau
import pybullet as p
from pybullet_utils import bullet_client as bullet_client
import pybullet_data as pd
import numpy as np

import time
from tqdm import tqdm
from motion_imitation.robots import mini_cheetah as mc
from motion_imitation.robots import robot_config
from locomotion_controller_mc import LocomotionController as Loco
from locomotion_controller_mc import Gait 
from swing_trajectory import FootSwingTrajectory as FST

GROUND_URDF_FILE = "plane.urdf"
DES_JNT_ANGLES = np.array(
    [0, -0.65,1.569, 0, -0.6, 1.8, 0, -0.65,1.569, 0, -0.6, 1.8])    
def main():    
    # create a pybullet client and connect to GUI server
    pybullet_client = bullet_client.BulletClient(p.GUI)
    # add pybullet_data to client search path
    pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    pybullet_client.setGravity(0,0,-10)
    # add ground
    pybullet_client.loadURDF(GROUND_URDF_FILE)
    # add robot
    robot = mc.MiniCheetah(pybullet_client=pybullet_client, action_repeat=1)
    
    # build a trotting gait
    timestep = 0.01
    gait = Gait(gait="trot",dt=timestep)    
    gait.setSwingDurations([0.1, 0.1, 0.1, 0.1]) # set swing phase durations in seconds
    gait.setStanceDurations([0.08, 0.08, 0.08, 0.08]) # set stance phase durations in seconds
    gait.setSwingTimeRemain([0,0,0,0]) # set remaining swing times to zero so that the robot starts in stance
    gait.setStanceTimeRemain([0.04, 0.08, 0.08, 0.04])

    # build swing trajectory generator
    fst = FST()
    fst.set_swing_height(0.05)

    # build trotting controller
    trot_ctrl = Loco(robot=robot, gait = gait, dt=timestep, swingTraj=fst)

    # desired trajectory
    vCoM_des = [0, 0, 0] # m/s
    quat_des = p.getQuaternionFromEuler([0,0,0])
    # run simulation
    robot.ReceiveObservation()
    len_sim = 5000
    for t in tqdm(range(len_sim)):
        print(t)
        quat = robot.GetBaseOrientation()
        pCoM = robot.GetBasePosition()
        w = robot.GetBaseVelocity()
        vCoM = robot.GetBaseVelocity()
        qJ = robot.GetMotorAngles()
        qJd = robot.GetMotorVelocities()        
        state = {'quat': quat, 
                 'pCoM': pCoM,
                 'omega': w,
                 'vCoM': vCoM,
                 'qJ': qJ,
                 'qJdot': qJd}          
        action = trot_ctrl.step(state, vCoM_des, quat_des)
        # action = np.zeros(12)
        robot.Step(action, control_mode=robot_config.MotorControlMode.TORQUE)
        time.sleep(timestep)

if __name__ == "__main__":
    main()
