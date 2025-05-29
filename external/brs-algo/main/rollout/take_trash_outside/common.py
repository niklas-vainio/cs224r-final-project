import numpy as np


INITIAL_QPOS = {
    "torso": np.array([0.028215, -0.07880333, -0.12412, 0.02912167]),
    "left_arm": np.array(
        [
            1.48734574e00,
            2.94314716e00,
            -2.58810284e00,
            -1.85106383e-03,
            -8.12056738e-04,
            2.91258865e-02,
        ]
    ),
    "right_arm": np.array(
        [-1.56693262, 2.84392553, -2.37629965, 0.0035922, 0.03647163, 0.01210816]
    ),
}


GRIPPER_CLOSE_STROKE = 0.5
GRIPPER_HALF_WIDTH = 50
NUM_PCD_POINTS = 4096
PAD_PCD_IF_LESS = True
PCD_X_RANGE = (0.0, 4.0)
PCD_Y_RANGE = (-1.0, 1.0)
PCD_Z_RANGE = (-0.5, 1.6)
MOBILE_BASE_VEL_ACTION_MIN = (-0.35, -0.35, -0.3)
MOBILE_BASE_VEL_ACTION_MAX = (0.35, 0.35, 0.3)
HORIZON_STEPS = 1300 * 4
CONTROL_FREQ = 100
ACTION_REPEAT = 12
