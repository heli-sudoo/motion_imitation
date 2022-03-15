import numpy as np


URDF_FILENAME = "mini_cheetah/mini_cheetah.urdf"
# mini_cheetah.urdf file could be found in pybullet examples
# https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data

REF_POS_SCALE = 0.75  # scale dog motion to quadruped of proper size
INIT_POS = np.array([0, 0, 0.3])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [
    7,  # left hand
    15,  # left foot
    3,  # right hand
    11,  # right foot    
]
SIM_HIP_JOINT_IDS = [4, 12, 0, 8]
SIM_ROOT_OFFSET = np.array([0, 0, -0.04]) # lower the trunk height to avoid weired leg configuration
SIM_TOE_OFFSET_LOCAL = [                  # offset the toe position of recorded dog motion
    np.array([0, 0.05, -0.01]),
    np.array([0, 0.05, 0.0]),
    np.array([0, -0.05, -0.01]),
    np.array([0, -0.05, 0.0])
]

# mini cheetah always start from this pose for all motions
# also used as initial guess to numerically solve IK
DEFAULT_JOINT_POSE = np.array(
    [0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])

# damping used by numerical IK solver
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])
