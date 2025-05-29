import argparse
import time
from common import (
    INITIAL_QPOS,
    GRIPPER_CLOSE_STROKE,
    NUM_PCD_POINTS,
    PAD_PCD_IF_LESS,
    PCD_X_RANGE,
    PCD_Y_RANGE,
    PCD_Z_RANGE,
    MOBILE_BASE_VEL_ACTION_MIN,
    MOBILE_BASE_VEL_ACTION_MAX,
    GRIPPER_HALF_WIDTH,
    HORIZON_STEPS,
    ACTION_REPEAT,
)
import numpy as np
import torch
from brs_ctrl.robot_interface import R1Interface
from brs_ctrl.robot_interface.grippers import GalaxeaR1G1Gripper
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import brs_algo.utils as U
from brs_algo.learning.policy import WBVIMAPolicy
from brs_algo.rollout.asynced_rollout import R1AsyncedRollout

DEVICE = torch.device("cuda:0")
NUM_LATEST_OBS = 2
HORIZON = 16
T_action_prediction = 8


def rollout(args):
    robot = R1Interface(
        left_gripper=GalaxeaR1G1Gripper(
            left_or_right="left", gripper_close_stroke=GRIPPER_CLOSE_STROKE
        ),
        right_gripper=GalaxeaR1G1Gripper(
            left_or_right="right", gripper_close_stroke=GRIPPER_CLOSE_STROKE
        ),
        enable_rgbd=False,
        enable_pointcloud=True,
        mobile_base_cmd_limit=np.array(MOBILE_BASE_VEL_ACTION_MAX),
        control_freq=65,
        wait_for_first_odom_msg=False,
    )

    policy = WBVIMAPolicy(
        prop_dim=21,
        prop_keys=[
            "odom/base_velocity",
            "qpos/torso",
            "qpos/left_arm",
            "qpos/left_gripper",
            "qpos/right_arm",
            "qpos/right_gripper",
        ],
        prop_mlp_hidden_depth=2,
        prop_mlp_hidden_dim=256,
        pointnet_n_coordinates=3,
        pointnet_n_color=3,
        pointnet_hidden_depth=2,
        pointnet_hidden_dim=256,
        action_keys=[
            "mobile_base",
            "torso",
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_key_dims={
            "mobile_base": 3,
            "torso": 4,
            "left_arm": 6,
            "left_gripper": 1,
            "right_arm": 6,
            "right_gripper": 1,
        },
        num_latest_obs=NUM_LATEST_OBS,
        use_modality_type_tokens=False,
        xf_n_embd=256,
        xf_n_layer=2,
        xf_n_head=8,
        xf_dropout_rate=0.1,
        xf_use_geglu=True,
        learnable_action_readout_token=False,
        action_dim=21,
        action_prediction_horizon=T_action_prediction,
        diffusion_step_embed_dim=128,
        unet_down_dims=[64, 128],
        unet_kernel_size=5,
        unet_n_groups=8,
        unet_cond_predict_scale=True,
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        ),
        noise_scheduler_step_kwargs=None,
        num_denoise_steps_per_inference=16,
    )
    U.load_state_dict(
        policy,
        U.torch_load(args.ckpt_path, map_location="cpu")["state_dict"],
        strip_prefix="policy.",
        strict=True,
    )
    policy = policy.to(DEVICE)
    policy.eval()

    rollout = R1AsyncedRollout(
        robot_interface=robot,
        num_pcd_points=NUM_PCD_POINTS,
        pcd_x_range=PCD_X_RANGE,
        pcd_y_range=PCD_Y_RANGE,
        pcd_z_range=PCD_Z_RANGE,
        mobile_base_vel_action_min=MOBILE_BASE_VEL_ACTION_MIN,
        mobile_base_vel_action_max=MOBILE_BASE_VEL_ACTION_MAX,
        gripper_half_width=GRIPPER_HALF_WIDTH,
        num_latest_obs=NUM_LATEST_OBS,
        num_deployed_actions=T_action_prediction,
        device=DEVICE,
        action_execute_start_idx=args.action_execute_start_idx,
        policy=policy,
        horizon_steps=HORIZON_STEPS,
        pad_pcd_if_needed=PAD_PCD_IF_LESS,
        action_repeat=ACTION_REPEAT,
    )

    input("Press [ENTER] to reset robot to initial qpos")
    # reset robot to initial qpos
    robot.control(
        arm_cmd={
            "left": INITIAL_QPOS["left_arm"],
            "right": INITIAL_QPOS["right_arm"],
        },
        gripper_cmd={
            "left": 0.1,
            "right": 0.1,
        },
        torso_cmd=INITIAL_QPOS["torso"],
    )

    input("Press [ENTER] to start rollout")
    for i in range(3):
        print(3 - i)
        time.sleep(1)
    rollout.rollout()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ckpt_path", type=str, required=True)
    args.add_argument("--action_execute_start_idx", type=int, default=1)
    args = args.parse_args()
    rollout(args)
