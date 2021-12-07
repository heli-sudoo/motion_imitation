import numpy as np
import pybullet as pb
import pybullet_data as pd
import time


def update_camera(robot):
  base_pos = np.array(pb.getBasePositionAndOrientation(robot)[0])
  [yaw, pitch, dist] = pb.getDebugVisualizerCamera()[8:11]
  pb.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
  return

def set_pose(robot, root_pos, root_quat, jnt_pose):
  num_joints = pb.getNumJoints(robot) 
  pb.resetBasePositionAndOrientation(robot, root_pos, root_quat)

  for j in range(num_joints):
    j_info = pb.getJointInfo(robot, j)
    j_state = pb.getJointStateMultiDof(robot, j)

    j_pose_idx = j_info[3]
    j_pose_size = len(j_state[0])
    j_vel_size = len(j_state[1])

    if (j_pose_size > 0):
      j_pose_idx = j_pose_idx - 7 # shift joint index by 7 to exclude base position and orientation
      j_pose = jnt_pose[j_pose_idx:(j_pose_idx + j_pose_size)]
      j_vel = np.zeros(j_vel_size)
      pb.resetJointStateMultiDof(robot, j, j_pose, j_vel)

  return

class ReplayData():
  def __init__(self, 
               robot_urdf,
               base_pos_traj,
               base_quat_traj,
               jnt_pos_traj):
    self.robot_urdf = robot_urdf
    self.base_pos_traj = base_pos_traj
    self.base_quat_traj = base_quat_traj
    self.jnt_pos_traj = jnt_pos_traj
            
def replay(replay_datas):
  pb.connect(pb.GUI,  options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")
  pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING,1)
  pb.setAdditionalSearchPath(pd.getDataPath())
  pb.setGravity(0,0,0)

  pb.loadURDF("plane.urdf")
  
  robots = []
  for i in range(len(replay_datas)):
    robots.append(pb.loadURDF(replay_datas[i].robot_urdf))

  
  num_frames = len(replay_datas[0].base_pos_traj)
  for f in range(num_frames):
    for i in range(len(replay_datas)):
      set_pose(robots[i], replay_datas[i].base_pos_traj[f], replay_datas[i].base_quat_traj[f], replay_datas[i].jnt_pos_traj[f])
      update_camera(robots[i])

    pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    time.sleep(0.1)

