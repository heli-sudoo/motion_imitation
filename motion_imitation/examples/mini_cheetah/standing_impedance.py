#!/bin/usr/python3

""" Joint PD control on simulated mini cheetah in Pybullet """

from numpy.lib.function_base import angle
import pybullet as p
from pybullet_utils import bullet_client as bullet_client
import pybullet_data as pd
import numpy as np

import time
from tqdm import tqdm
from motion_imitation.robots import mini_cheetah as mc
from motion_imitation.robots import robot_config
from locomotion_controller_mc import leg_impedance_control as leg_imp_ctrl

DES_JNT_ANGLES = np.array(
    [0., -.8, 1.6, 0., -.8, 1.6, 0., -.8, 1.6, 0., -.8, 1.6])  
def main():    
    # create a pybullet client and connect to GUI server
    pybullet_client = bullet_client.BulletClient(p.GUI)
    # add pybullet_data to client search path
    pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    pybullet_client.setGravity(0,0,-2)
    pybullet_client.loadURDF("plane.urdf")
    # pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,10)        

    robot = mc.MiniCheetah(pybullet_client=pybullet_client, action_repeat=1, time_step=0.001)
    # udpate the observation
    robot.ReceiveObservation()
    init_jnt_angles = np.reshape(robot.GetTrueMotorAngles(), [4, 3])
    # init_foot_pos = [robot.computeFootPositionInHip(init_jnt_angles[leg_id], leg_id) for leg_id in range(4)]
    # init_foot_pos = np.reshape(init_foot_pos, [4, 3])
    height_des = 0.3 # desired height
   

    len_sim = 50000
    for t in tqdm(range(len_sim)):
        print(t)        
        jnt_angles = np.reshape(robot.GetTrueMotorAngles(), [4, 3])
        jnt_vels = np.reshape(robot.GetTrueMotorVelocities(), [4, 3])  
        if t < 500:
            progress = min(t/300, 1)
            des_jnt_angles = (1-progress)*init_jnt_angles.reshape(12) + progress * DES_JNT_ANGLES
            action = des_jnt_angles
            robot.Step(action, control_mode=robot_config.MotorControlMode.POSITION)            
            init_foot_pos = [robot.computeFootPositionInHip(jnt_angles[leg_id], leg_id) for leg_id in range(4)]
            init_foot_pos = np.reshape(init_foot_pos, [4, 3])
        else:
            progress = min((t-500)/500, 1)                      
            # impedance control
            tau = np.zeros([4,3])
            for leg_id in range(4):
                # des_foot_pos = init_foot_pos[leg_id]
                des_foot_pos = np.array([.0,.0,.0])
                des_foot_pos[2] = -height_des
                # print(des_foot_pos)
                cmd_foot_pos = (1-progress)*init_foot_pos[leg_id] + progress*des_foot_pos
                leg_Jacob = robot.computeLegJacobian(jnt_angles[leg_id], leg_id)
                foot_pos = robot.computeFootPositionInHip(jnt_angles[leg_id], leg_id)
                foot_vel = robot.computeFootVelInHip(jnt_angles[leg_id], jnt_vels[leg_id], leg_id)
                cmd_foot_vel = np.zeros(3)
                tau[leg_id] = leg_imp_ctrl(leg_Jacob, cmd_foot_pos, foot_pos, cmd_foot_vel, foot_vel)
            action = np.reshape(tau, 12)
            robot.Step(action, control_mode=robot_config.MotorControlMode.TORQUE)
        time.sleep(0.001)

if __name__ == "__main__":
    main()
