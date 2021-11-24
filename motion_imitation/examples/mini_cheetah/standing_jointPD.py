#!/bin/usr/python3

""" Joint PD control on simulated mini cheetah in Pybullet """

import pybullet as p
from pybullet_utils import bullet_client as bullet_client
import pybullet_data as pd
import numpy as np

import time
from tqdm import tqdm
from motion_imitation.robots import mini_cheetah as mc
from motion_imitation.robots import robot_config

GROUND_URDF_FILE = "plane.urdf"
DES_JNT_ANGLES = np.array(
    [0., -.8, 1.6, 0., -.8, 1.6, 0., -.8, 1.6, 0., -.8, 1.6])    
def main():    
    # create a pybullet client and connect to GUI server
    pybullet_client = bullet_client.BulletClient(p.GUI)
    # add pybullet_data to client search path
    pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    pybullet_client.setGravity(0,0,-10)
    pybullet_client.loadURDF(GROUND_URDF_FILE)
    # pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,10)        

    robot = mc.MiniCheetah(pybullet_client=pybullet_client, action_repeat=1)
    # udpate the observation
    robot.ReceiveObservation()
    init_jnt_angles = robot.GetTrueMotorAngles() # in nparray

    len_sim = 5000
    for t in tqdm(range(len_sim)):
        print(t)
        progress = min(t/300, 1)
        des_jnt_angles = (1-progress)*init_jnt_angles + progress * DES_JNT_ANGLES
        action = des_jnt_angles
        robot.Step(action, control_mode=robot_config.MotorControlMode.POSITION)
        # pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        time.sleep(0.01)

if __name__ == "__main__":
    main()
