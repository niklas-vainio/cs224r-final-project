# Class to wrap the WB-VIMA policy, exposing a query_action method for inference

from brs_algo.learning.policy import WBVIMAPolicy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import brs_algo.utils as U

from collections import deque

import torch as th
import numpy as np

DEVICE = th.device("cuda:0")


class WBVIMAPolicyWrapper():

    # Joint limits for R1 pro: can be exported from sim with robot.joint_lower_limits
    # and robot.joint_upper_limits (refer to indices in robots.joints)
    torso_joint_high = np.array([1.8326, 2.5307, 1.5708, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])

    left_arm_joint_high = np.array([1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708])
    left_arm_joint_low = np.array([-4.4506, -0.1745, -2.3562, -1.7453, -2.3562, -1.0472, -1.5708])

    right_arm_joint_high = np.array([1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708])
    right_arm_joint_low = np.array([-4.4506, -3.1416, -2.3562, -1.7453, -2.3562, -1.0472, -1.5708])

    pcd_xyz_min = np.array([-5.0, -5.0, -5.0])
    pcd_xyz_max = np.array([5.0, 5.0, 5.0])

    mobile_base_vel_action_min = np.array([-0.3, -0.3, -0.4])
    mobile_base_vel_action_max = np.array([0.3, 0.3, 0.4])

    NUM_LATEST_OBS = 2
    ACTION_PREDICTION_HORIZON = 8
    NUM_DEPLOYED_ACTIONS = 8

    def __init__(self, checkpoint_path: str):
        self.obs_history = deque(maxlen=self.NUM_LATEST_OBS)
        self.action_idx = 0

        # Wrap internal policy - this should exactly match training config
        self._policy = WBVIMAPolicy(
            prop_dim=68,
            prop_keys=[
                "proprio/all",
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
                "left_arm": 7,
                "left_gripper": 1,
                "right_arm": 7,
                "right_gripper": 1,
            },
            num_latest_obs=self.NUM_LATEST_OBS,
            use_modality_type_tokens=False,
            xf_n_embd=256,
            xf_n_layer=2,
            xf_n_head=8,
            xf_dropout_rate=0.1,
            xf_use_geglu=True,
            learnable_action_readout_token=False,
            action_dim=21,
            action_prediction_horizon=self.ACTION_PREDICTION_HORIZON,
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
            U.torch_load(checkpoint_path, map_location="cpu")["state_dict"],
            strip_prefix="policy.",
            strict=True,
        )
        policy = policy.to(DEVICE)
        policy.eval()

    def reset(self):
        """
        Clear observation history and reset action index
        """
        self.obs_history.clear()
        self.action_idx = 0

    def query_action(self, _obs: dict) -> th.Tensor:
        """
        Run inference to get the action for the given observation
        """

        # Add normalized obs to history
        if len(self.obs_history) == 0:
            for _ in range(self.NUM_LATEST_OBS):
                self.obs_history.append(self._normalize_obs(_obs))
            else:
                self.obs_history.append(self._normalize_obs(_obs))

        obs = U.any_concat(self.obs_history, dim=1)  # (B = 1, T = num_latest_obs, ...)

        # Run inference every NUM_DEPLOYED_ACTIONS timesteps
        if self.action_idx % self.NUM_DEPLOYED_ACTIONS == 0:
            action_traj_pred = self._policy.act(obs)
            self.action_traj_pred = {
                k: v[0].detach().cpu().numpy() for k, v in action_traj_pred.items()
            }  # dict of (T_A, ...)
            self.action_idx = 0

        self.action_idx += 1
        action = U.any_slice(action_traj_pred, np.s_[self.action_idx])

        # Un-normalize the action so it arrives in the format expected by the sim
        return self._unnormalize_action(action)

    def _normalize_obs(self, obs: dict) -> dict:
        """
        Normalize observation dict from sim to be compatible with the policy
        """
        pcd = obs["pointcloud"]
        all_proprio = obs["proprio"]["all"]

        pcd_xyz = (
            2
            * (pcd["xyz"] - self.pcd_xyz_min)
            / (self.pcd_xyz_max - self.pcd_xyz_min)
            - 1
        ).astype(np.float32)
        pcd_rgb = (pcd["rgb"] / 255).astype(np.float32)

        obs_dict = {
            "pointcloud": {
                "xyz": th.tensor(
                    pcd_xyz, device=DEVICE, dtype=th.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),  # (B=1, T=1, N, 3)
                "rgb": th.tensor(
                    pcd_rgb, device=DEVICE, dtype=th.float32
                )
                .unsqueeze(0)
                .unsqueeze(0), # (B=1, T=1, N, 3)
            },
            "proprio": {
                "all": th.tensor(
                    all_proprio, device=DEVICE, dtype=th.float32
                )
                .unsqueeze(0)
                .unsqueeze(0)
            }
        }
        return obs_dict

    def _unnormalize_action(
        self, action: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Un-normalize an output from the policy to make it compatible with the simulation
        """
        mobile_base_vel_cmd = action["mobile_base"]
        mobile_base_vel_cmd = np.clip(mobile_base_vel_cmd, -1, 1)
        
        mobile_base_vel_cmd = (mobile_base_vel_cmd + 1) / 2 * (
            self.mobile_base_vel_action_max - self.mobile_base_vel_action_min
        ) + self.mobile_base_vel_action_min

        left_arm = action["left_arm"]
        left_arm = (left_arm + 1) / 2 * (
            self.left_arm_joint_high - self.left_arm_joint_low
        ) + self.left_arm_joint_low

        right_arm = action["right_arm"]
        right_arm = (right_arm + 1) / 2 * (
            self.right_arm_joint_high - self.right_arm_joint_low
        ) + self.right_arm_joint_low

        torso = action["torso"]
        torso = (torso + 1) / 2 * (
            self.torso_joint_high - self.torso_joint_low
        ) + self.torso_joint_low

        # Use gripper actions directly
        left_gripper = action["left_gripper"]
        right_gripper = action["right_gripper"]

        # Squeeze into torch tensor, so this is compatible with the sim
        return th.concat((
            mobile_base_vel_cmd,
            torso,
            left_arm,
            left_gripper,
            right_arm,
            right_gripper
        )) # Shape should be be (23)
