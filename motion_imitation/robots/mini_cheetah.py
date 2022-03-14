#!/bin/usr/python3

"""Pybullet simulation of mini cheetah robot."""
import math
import os

import numpy as np
import re
import pybullet
import pybullet_data as pd
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.robots import mini_cheetah_pose_utils as mc_pose
from motion_imitation.envs import locomotion_gym_config

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
HIP_NAME_PATTERN = re.compile(r"\w+_to_abduct_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_to_thigh_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_to_knee_\w+")
TOE_NAME_PATTERN = re.compile(r"toe_\w+")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2

DEFAULT_TORQUE_LIMITS = [12, 18, 12] * 4
INIT_RACK_POSITION = [0, 0, 1.4]
INIT_POSITION = [0, 0, 0.404]
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array([0.0, 0.0, 0.0] * 4)
PI = math.pi
ABAD_LOCATIONS = np.array([[0.19, 0.049, 0],
                          [-0.19, 0.049, 0],
                          [0.19, -0.049, 0],
                          [-0.19, -0.049, 0]])

ABAD_UPPER_BOUND = 2*PI/3
HIP_UPPER_BOUND = 1.5*PI
KNEE_UPPER_BOUND = 0.86*PI

# Bases on the readings from 's default pose.
INIT_MOTOR_ANGLES = [
    mc_pose.DEFAULT_ABDUCTION_ANGLE, 
    mc_pose.DEFAULT_HIP_ANGLE, 
    mc_pose.DEFAULT_KNEE_ANGLE
] * NUM_LEGS

ABDUCTION_P_GAIN = 80.0
ABDUCTION_D_GAIN = 1.
HIP_P_GAIN = 80.0
HIP_D_GAIN = 1.0
KNEE_P_GAIN = 80.0
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
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="FL_hip_motor",
                                        upper_bound=ABAD_UPPER_BOUND,
                                        lower_bound=-ABAD_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="FL_upper_joint",
                                        upper_bound=HIP_UPPER_BOUND,
                                        lower_bound=-HIP_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="FL_lower_joint",
                                        upper_bound=KNEE_UPPER_BOUND,
                                        lower_bound=-KNEE_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="RL_hip_motor",
                                        upper_bound=ABAD_UPPER_BOUND,
                                        lower_bound=-ABAD_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="RL_upper_joint",
                                        upper_bound=HIP_UPPER_BOUND,
                                        lower_bound=-HIP_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="RL_lower_joint",
                                        upper_bound=KNEE_UPPER_BOUND,
                                        lower_bound=-KNEE_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="FR_hip_motor",
                                        upper_bound=ABAD_UPPER_BOUND,
                                        lower_bound=-ABAD_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="FR_upper_joint",
                                        upper_bound=HIP_UPPER_BOUND,
                                        lower_bound=-HIP_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="FR_lower_joint",
                                        upper_bound=KNEE_UPPER_BOUND,
                                        lower_bound=-KNEE_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="RR_hip_motor",
                                        upper_bound=ABAD_UPPER_BOUND,
                                        lower_bound=-ABAD_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="RR_upper_joint",
                                        upper_bound=HIP_UPPER_BOUND,
                                        lower_bound=-HIP_UPPER_BOUND),
      locomotion_gym_config.ScalarField(name="RR_lower_joint",
                                        upper_bound=-KNEE_UPPER_BOUND,
                                        lower_bound=-KNEE_UPPER_BOUND),
  ]
  def __init__( self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      sensors=None,
      control_latency=0.002,
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
    # change the appearance color
    robot_visuals = self._pybullet_client.getVisualShapeData(self.quadruped)
    # change visual color of the robot
    for visual_obj in robot_visuals:
        if visual_obj[1] not in [3,7,11,15]:
            self._pybullet_client.changeVisualShape(self.quadruped, visual_obj[1], rgbaColor = [0.65, 0.65, 0.65, 1])

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
    return self._urdf_filename
  
  def GetLowerLinkIDs(self):
      return self._lower_link_ids

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
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)
    self._hip_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._lower_link_ids = []
    self._foot_link_ids = []
    self._imu_link_ids = []

    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if HIP_NAME_PATTERN.match(joint_name):
        self._hip_link_ids.append(joint_id)
      elif UPPER_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif LOWER_NAME_PATTERN.match(joint_name):
        self._lower_link_ids.append(joint_id)
      elif TOE_NAME_PATTERN.match(joint_name):
        #assert self._urdf_filename == URDF_WITH_TOES
        self._foot_link_ids.append(joint_id)
      elif IMU_NAME_PATTERN.match(joint_name):
        self._imu_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)
    self._leg_link_ids.extend(self._lower_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)

    #assert len(self._foot_link_ids) == NUM_LEGS
    self._hip_link_ids.sort()
    self._motor_link_ids.sort()
    self._lower_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    else:
      return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    init_orientation = pybullet.getQuaternionFromEuler([0., 0., 0.])
    return init_orientation
  
  def _ClipMotorCommands(self, motor_commands):
    """Clips motor commands.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).

    Returns:
      Clipped motor commands.
    """

    # clamp the motor command by the joint limit, in case weired things happens
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    motor_commands = np.clip(motor_commands,
                             current_motor_angles - max_angle_change,
                             current_motor_angles + max_angle_change)
    return motor_commands

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose
  
  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).N
      motor_control_mode: A MotorControlMode enum.
    """
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)
    super(MiniCheetah, self).ApplyAction(motor_commands, motor_control_mode)

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
 
