import pybullet
import time
import pybullet_data as pd
import numpy as np
import time

POS_SIZE = 3
ROT_SIZE = 4
INIT_POS = np.array([0, 0, 0.32])
INIT_ROT = pybullet.getQuaternionFromEuler([0,0,0])
DEFAULT_JOINT_POSE = np.zeros(12)
DEFAULT_POSE = np.concatenate([INIT_POS, INIT_ROT, DEFAULT_JOINT_POSE])

def get_root_pos(pose):
  return pose[0:POS_SIZE]

def get_root_rot(pose):
  return pose[POS_SIZE:(POS_SIZE + ROT_SIZE)]

def get_joint_pose(pose):
  return pose[(POS_SIZE + ROT_SIZE):]

def update_camera(robot):
  base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
  [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
  pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
  return

def set_pose(robot, pose):
  num_joints = pybullet.getNumJoints(robot)
  root_pos = get_root_pos(pose)
  root_rot = get_root_rot(pose)
  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  for j in range(num_joints):
    j_info = pybullet.getJointInfo(robot, j)
    j_state = pybullet.getJointStateMultiDof(robot, j)

    j_pose_idx = j_info[3]
    j_pose_size = len(j_state[0])
    j_vel_size = len(j_state[1])

    if (j_pose_size > 0):
      j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
      j_vel = np.zeros(j_vel_size)
      pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)

  return

def plotA1Config(pose = DEFAULT_JOINT_POSE):
  p = pybullet
  p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

  pybullet.setAdditionalSearchPath(pd.getDataPath())


  planeId = pybullet.loadURDF("plane.urdf")
  robot = pybullet.loadURDF("a1/a1.urdf", INIT_POS, INIT_ROT)

  while True:
      set_pose(robot, pose)
      update_camera(robot)
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
      time.sleep(1./240.)
  p.disconnect()
  return


