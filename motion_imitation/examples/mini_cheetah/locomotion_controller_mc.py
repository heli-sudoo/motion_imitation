#!/bin/usr/python3
""" This file implements trotting gait on mini cheetah using impedance control and Raibert
    heuristics """

import numpy as np
from motion_imitation.robots import mini_cheetah as mc
from motion_imitation.robots.mini_cheetah import MiniCheetah as MiniCheetah
from swing_trajectory import FootSwingTrajectory as FST
import pybullet as pb

ABAD_GAIN = 50.
ABAD_D = 1.
HIP_GAIN = 50.
HIP_D = 1.
KNEE_GAIN = 50.
KNEE_D = 1.
KP_LEG = np.diag([ABAD_GAIN, HIP_GAIN, KNEE_GAIN])
KD_LEG = np.diag([ABAD_D, HIP_D, KNEE_D])

GAIT_LIB = ["trot", "pronk", "bound"]
class Gait():
    def __init__(self, gait="trot", dt = 0.001):
        if gait not in GAIT_LIB:
            print("Input gait is not achievable")
            print("Use default trotting gait")
            self.gait = "trot"
        self.gait = gait        
        self.dt = dt
        self.swingDuration = (np.array([0.1] * 4)/dt).astype(int)
        self.stanceDuration = (np.array([0.08] * 4)/dt).astype(int)
        self.swingTimeRemain = (np.array([0.1] * 4)/dt).astype(int)
        self.stanceTimeRemain = (np.array([0.08] * 4)/dt).astype(int)
        self.swingProgress = np.array([0] * 4)
        self.stanceProgress = np.array([0] * 4)  
        self.footStatus = ["c", "c", "c", "c"] # c means contact, s means swing      

    def step(self):
        """ update the status of each leg 
            time: current time
        """
        for leg in range(4):
            if self.footStatus[leg] == "c":
                if self.stanceTimeRemain[leg] > 0:
                    self.stanceTimeRemain[leg] -= 1    
                else: # stance finished
                    self.stanceProgress[leg] = 1
                    self.footStatus[leg] = "s"
                    self.swingProgress[leg] = 0
                    continue
            if self.footStatus[leg] == "s":
                if self.swingTimeRemain[leg] > 0:
                    self.swingTimeRemain[leg] -= 1
                else:
                    self.swingProgress[leg] = 1
                    self.footStatus[leg] = "c"
                    self.stanceProgress[leg] = 0
                    continue
            # update swing and stance progress
            self.swingProgress[leg] = (self.swingDuration[leg] - self.swingTimeRemain[leg]) / self.swingDuration[leg]
            self.stanceProgress[leg] = (self.stanceDuration[leg] - self.stanceTimeRemain[leg]) / self.stanceDuration[leg]                                                    

    def getSwingProgress(self):
        return self.swingProgress

    def getStanceProgress(self):
        return self.stanceProgress
    
    def getFootStatus(self):
        return self.footStatus
    
    def setSwingDurations(self, swingDurs):
        self.swingDuration = (np.asarray(swingDurs)/ self.dt).astype(int)
        self.updateSwingProgress()
    
    def setStanceDurations(self, stanceDurs):
        self.stanceDuration = (np.asarray(stanceDurs)/ self.dt).astype(int)
        self.udpateStanceProgress()

    def setSwingTimeRemain(self, remSwingTime):
        assert len(remSwingTime) == 4, "swing time should have length 4"
        self.swingTimeRemain = (np.asarray(remSwingTime)/ self.dt).astype(int)
        self.updateSwingProgress()
    
    def setStanceTimeRemain(self, remStanceTime):
        assert len(remStanceTime) == 4, "stance time should have length 4"        
        self.stanceTimeRemain = (np.asarray(remStanceTime)/ self.dt).astype(int)
        self.udpateStanceProgress()
    
    def updateSwingProgress(self):
        self.swingProgress = (self.swingDuration - self.swingTimeRemain) / self.swingDuration
    
    def udpateStanceProgress(self):
        self.stanceProgress = (self.stanceDuration - self.stanceTimeRemain) / self.stanceDuration

def leg_impedance_control(J, pd, p, vd, v):
    kp = np.diag([800.,800.,800.])
    kd = np.diag([10.,10.,10.])
    tau = J.transpose() @ (kp@ np.transpose(pd-p) + kd@ np.transpose(vd - v))
    return tau.transpose()

class LocomotionController:
    def __init__(self, 
                 robot=MiniCheetah,
                 gait = Gait,
                 dt = 0.001,
                 swingTraj = FST):
        self.robot = robot
        self.gait = gait
        self.dt = dt
        self.swingTraj = swingTraj

        self.pCoM = np.zeros(3)
        self.vCoM = np.zeros(3)
        self.quat = np.array([1,0,0,0])
        self.w = np.zeros(3)
        self.qJ = np.zeros([4,3])
        self.qJd = np.zeros([4,3])
        
        self.iter = 0

        # foot positions in global frame
        self.footPositions = np.zeros([4,3])
        # hip positions in global frame
        self.hipPositions = np.zeros([4,3])
        # foothold locations in global frame    
        self.footHoldLocPrev = np.zeros([4,3]) 
        self.footHoldLocNext = np.zeros([4,3])
        # stance foot target
        self.stanceFootPosTargets = np.zeros([4,3])
        # foot status
        self.footStatusPrev = ["c","c","c","c"]
        self.footStatusCurr = ["c","c","c","c"]
        # swing leg progress
        self.swingProgress = np.zeros(4)

    def stanceLegControl(self, vCoM_des, leg):
        R = np.reshape(pb.getMatrixFromQuaternion(self.quat),[3,3])
        stanceFootPos = self.robot.computeFootPositionInHip(self.qJ[leg], leg)
        if self.footStatusPrev[leg] == "s":
            self.stanceFootPosTargets[leg] = stanceFootPos
        else:
            self.stanceFootPosTargets[leg] -= vCoM_des * self.dt 
        stanceFootVelTarget = np.transpose(-R @ vCoM_des.transpose())
        stanceFootVel = self.robot.computeFootVelInHip(self.qJ[leg], self.qJd[leg], leg)
        self.impedance_control_leg(self.stanceFootPosTargets[leg], stanceFootPos, stanceFootVelTarget, stanceFootVel, leg)

    def swingLegControl(self, leg):
        self.swingTraj.set_start_location(self.footHoldLocPrev[leg])
        self.swingTraj.set_end_location(self.footHoldLocNext[leg])
        self.swingTraj.computeTrajectory(self.swingProgress[leg], self.gait.swingDuration[leg])
        pfoot_des_global = self.swingTraj.get_position()
        vfoot_des_global = self.swingTraj.get_velocity()
        R = np.reshape(pb.getMatrixFromQuaternion(self.quat),[3,3])
        pfoot_des_hip = np.transpose(R@ np.transpose(pfoot_des_global - self.hipPositions[leg]))
        vfoot_des_hip = np.transpose(R@ np.transpose(vfoot_des_global - self.vCoM))
        pfoot_hip = self.robot.computeFootPositionInHip(self.qJ[leg], leg)
        vfoot_hip = self.robot.computeFootVelInHip(self.qJ[leg], self.qJd[leg], leg)
        self.impedance_control_leg(pfoot_des_hip, pfoot_hip, vfoot_des_hip, vfoot_hip, leg)

    def step(self, state, vCoM_des, quat_des = pb.getQuaternionFromEuler([0,0,0])):
        """ compute control for the current state
        Input @state: robot state in dictionary
              @vCoM_des: desired CoM velocity
              @quat_des: desired body orientation in quaternion
        Return 
              @tau: joint torques in np.array of dimension 12
        """
        self.iter += 1

        self.pCoM = state["pCoM"]
        self.quat = state["quat"]
        self.vCoM = state["vCoM"]
        self.w    = state["omega"]
        self.qJ   = np.reshape(state["qJ"], [4, 3])
        self.qJd  = np.reshape(state["qJdot"], [4, 3])
        
        self.updateFootPositions()
        self.updateHipPositions()
        self.updateFootStatus()
        self.updateFootHoldLocations(vCoM_des)
        self.updateSwingLegProgress()

        vCoM_des = np.asarray(vCoM_des)
        tau = np.zeros([4,3])
        for leg in range(4):
            if self.footStatusCurr[leg] == "c":
                tau[leg] = self.stanceLegControl(vCoM_des, leg)
            if self.footStatusCurr[leg] == "s":
                tau[leg] = self.swingLegControl(leg)
        self.gait.step()
        return np.reshape(tau, 12)
        
    def updateFootPositions(self):
        """ update foot positions in global frame"""
        self.footPositions = self.robot.computeFootPositionsInGlobal(self.quat, self.pCoM, self.qJ)
    
    def updateHipPositions(self):
        """ update the hip positions in global frame"""
        self.hipPositions = self.robot.computeHipPositionsInGlobal(self.quat, self.pCoM)

    def updateFootStatus(self):
        if self.iter == 1:
            self.footStatusCurr = self.gait.getFootStatus()
        self.footStatusPrev = self.footStatusCurr
        self.footStatusCurr = self.gait.getFootStatus()
    
    def updateFootHoldLocations(self, vCoM_des):
        for leg in range(4):
            if self.footStatusCurr[leg] == "c":
                # set the previous foothold location to the newest foot position during stance
                self.footHoldLocPrev[leg] = self.footPositions[leg]
            if self.footStatusPrev[leg] == "c" and  self.footStatusCurr[leg] == "s":
                # set the predicted foothold location when transitioning from stance to swing
                vCoM = self.vCoM
                vCoM[2] = 0
                vCoM_des[2] = 0
                pos_hip = self.hipPositions[leg]                
                self.footHoldLocPrev[leg] = pos_hip + vCoM_des * self.gait.swingDuration[leg]/2
                k = 1.0
                self.footHoldLocPrev[leg] += k * (vCoM_des - vCoM)* self.gait.swingDuration[leg]

    def updateSwingLegProgress(self):
        self.swingProgress = self.gait.swingProgress
    
    def impedance_control_leg(self, pd, p, vd, v, leg):
        """ Impedance control for a single leg in fixed abad frame
        input 
            pd: desired foot position in hip frame
            p: foot position in hip frame
            v: foot velocity in hip frame    
        """
        J = self.robot.computeLegJacobian(self.qJ[leg], leg)
        return leg_impedance_control(J,pd,p,vd,v)
     