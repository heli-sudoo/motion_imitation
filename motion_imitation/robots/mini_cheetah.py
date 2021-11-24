#!/bin/usr/python3

"""Pybullet simulation of mini cheetah robot."""
import math
import os

import numpy as np
import pybullet
import pybullet_data as pd
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config

URDF_FILENAME = "mini_cheetah/mini_cheetah.urdf"

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "torso_to_abduct_fl_j",  # Left front abduction (hip0).
    "abduct_fl_to_thigh_fl_j",  # Left front hip (upper0).
    "thigh_fl_to_knee_fl_j",  # Left front knee (lower0).
    "torso_to_abduct_hl_j",  # Left rear abduction (hip1).
    "abduct_hl_to_thigh_hl_j",  # Left rear hip (upper1).
    "thigh_hl_to_knee_hl_j",  # Left rear knee (lower1).
    "torso_to_abduct_fr_j",  # Right front abduction (hip2).
    "abduct_fr_to_thigh_fr_j",  # Right front hip (upper2).
    "thigh_fr_to_knee_fr_j",  # Right front knee (lower2).
    "torso_to_abduct_hr_j",  # Right rear abduction (hip3).
    "abduct_hr_to_thigh_hr_j",  # Right rear hip (upper3).
    "thigh_hr_to_knee_hr_j",  # Right rear knee (lower3).
]
DEFAULT_TORQUE_LIMITS = [12, 18, 12] * 4
INIT_RACK_POSITION = [0, 0, 1.4]
INIT_POSITION = [0, 0, 0.4]
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array([0.0, 0.0, 0.0] * 4)
PI = math.pi
ABAD_LOCATIONS = np.array([[0.19, 0.049, 0],
                          [-0.19, 0.049, 0],
                          [0.19, -0.049, 0],
                          [-0.19, -0.049, 0]])
DEFAULT_ABDUCTION_ANGLE = 0.0
DEFAULT_HIP_ANGLE = -1.1
DEFAULT_KNEE_ANGLE = 2.3
# Bases on the readings from 's default pose.
INIT_MOTOR_ANGLES = [
    DEFAULT_ABDUCTION_ANGLE, DEFAULT_HIP_ANGLE, DEFAULT_KNEE_ANGLE
] * NUM_LEGS


# left hand, left foot, right hand, right foot
ABAD_LINK_IDS = [5, 13, 1, 9]
THEIGH_LINK_IDS = [6, 14, 2, 10]
SHANK_LINK_IDS = [7, 15, 3, 11]
TOE_LINK_IDS = [8, 16, 4, 12]

ABDUCTION_P_GAIN = 50.0
ABDUCTION_D_GAIN = 1.
HIP_P_GAIN = 50.0
HIP_D_GAIN = 1.0
KNEE_P_GAIN = 50.0
KNEE_D_GAIN = 1.0

ABAD_LINK_LENGTH = 0.062
HIP_LINK_LENGTH = 0.209
KNEE_LINK_LENGTH = 0.195
KNEE_LINK_YOFFSET = 0.004

def getSideSign(leg_id):
  """Get if the leg is on the right (-1) or the left (+) of the robot"""
  sideSigns = [1,1,-1,-1]
  return sideSigns[leg_id]

def analytical_leg_jacobian(leg_angles, leg_id):
  l1 = ABAD_LINK_LENGTH
  l2 = HIP_LINK_LENGTH
  l3 = KNEE_LINK_LENGTH
  l4 = KNEE_LINK_YOFFSET 
  sideSign = getSideSign(leg_id)

  s1 = math.sin(leg_angles[0])
  s2 = math.sin(leg_angles[1])
  s3 = math.sin(leg_angles[2])

  c1 = math.cos(leg_angles[0])
  c2 = math.cos(leg_angles[1])
  c3 = math.cos(leg_angles[2])

  c23 = c2*c3 - s2*s3
  s23 = s2*c3 + c2*s3

  J = np.zeros([3, 3])
  J[0,1] = l3 * c23 + l2 * c2
  J[0,2] = l3 * c23
  J[1,0] = l3 * c1 * c23 + l2 * c1 * c2 - (l1+l4) * sideSign * s1
  J[1,1] = -l3 * s1 * s23 - l2 * s1 * s2
  J[1,2] = -l3 * s1 * s23
  J[2,0] = l3 * s1 * c23 + l2 * c2 * s1 + (l1+l4) * sideSign * c1
  J[2,1] = l3 * c1 * s23 + l2 * c1 * s2
  J[2,2] = l3 * c1 * s23

  return J

def foot_position_in_hip(leg_angles, leg_id):
  l1 = ABAD_LINK_LENGTH
  l2 = HIP_LINK_LENGTH
  l3 = KNEE_LINK_LENGTH
  l4 = KNEE_LINK_YOFFSET 
  sideSign = getSideSign(leg_id)

  s1 = math.sin(leg_angles[0])
  s2 = math.sin(leg_angles[1])
  s3 = math.sin(leg_angles[2])

  c1 = math.cos(leg_angles[0])
  c2 = math.cos(leg_angles[1])
  c3 = math.cos(leg_angles[2])

  c23 = c2*c3 - s2*s3
  s23 = s2*c3 + c2*s3

  p = np.zeros(3)
  p[0] = l3 * s23 + l2 * s2
  p[1] = (l1+l4) * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
  p[2] = (l1+l4) * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

  return p

def foot_positions_in_body(leg_angles):
  leg_angles = leg_angles.reshape((4, 3))
  foot_positions = np.zeros((4, 3))
  for leg in range(4):
    foot_positions[leg] = foot_position_in_hip(leg_angles[leg], leg)
  return foot_positions + ABAD_LOCATIONS

class MiniCheetah(minitaur.Minitaur):
  """A simulation for the mini cheetah robot."""

  def __init__( self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      sensors=None,
      control_latency=0.001,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=None,
      reset_time=1,
      allow_knee_contact=False):

    self._urdf_filename = urdf_filename
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands

    motor_kp = [
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
        HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
    ]
    motor_kd = [
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
        HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
    ]
    super(MiniCheetah, self).__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_control_mode=motor_control_mode,
        motor_model_class=laikago_motor.LaikagoMotorModel,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        on_rack=on_rack,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter,
        reset_time=reset_time)

  def _LoadRobotURDF(self):
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          self._urdf_filename,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          self._urdf_filename, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()
    for _ in range(500):
      self.ApplyAction(
          INIT_MOTOR_ANGLES,
          motor_control_mode=robot_config.MotorControlMode.POSITION)
      self._pybullet_client.stepSimulation()
      self.ReceiveObservation()
    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self.ApplyAction(
            default_motor_angles,
            motor_control_mode=robot_config.MotorControlMode.POSITION)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()

  def GetURDFFile(self):
    return os.path.join(self._urdf_root, "mini_cheetah/mini_cheetah.urdf")

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      angle = INIT_MOTOR_ANGLES[i]
      self._pybullet_client.resetJointState(
          self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

  def _BuildUrdfIds(self):
    pass

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    else:
      return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    init_orientation = [0, 0, 0, 1.0]
    return init_orientation
  
  def computeLegJacobian(self, leg_angle, leg_id):
    return analytical_leg_jacobian(leg_angle, leg_id)
  
  def computeFootPositionsInBody(self, leg_angles):
    return foot_positions_in_body(leg_angles)
  
  def computeFootPositionInHip(self,leg_angle, leg_id):
    return foot_position_in_hip(leg_angle, leg_id)
  
  def computeHipPositionsInGlobal(self, quat, pos):
    R = np.reshape(pybullet.getMatrixFromQuaternion(quat),[3,3])
    return pos + (R @ ABAD_LOCATIONS.transpose()).transpose()
  
  def computeFootVelInHip(self, leg_angle, leg_vel, leg):
    J = self.computeLegJacobian(leg_angle, leg)
    return np.transpose(J@ leg_vel.transpose())

  def computeFootPositionsInGlobal(self, quat, pos, leg_angles):
    foot_pos_body = self.computeFootPositionsInBody(leg_angles)
    R = np.reshape(pybullet.getMatrixFromQuaternion(quat),[3,3])
    foot_pos_global = R @ np.transpose(foot_pos_body + pos)
    return foot_pos_global.transpose()
 
