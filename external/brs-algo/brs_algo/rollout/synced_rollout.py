from typing import Union, Optional, Tuple, Dict, List

import os
import re
from collections import deque

import torch
import numpy as np
import rospy
import pybullet as pb
from brs_ctrl.asset_root import ASSET_ROOT
from brs_ctrl.robot_interface import R1Interface
from tqdm import tqdm
from einops import rearrange

from brs_algo.utils import any_concat, any_to_torch_tensor, any_slice
from brs_algo.learning.policy.base import BaseDiffusionPolicy


class R1SyncedRollout:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])

    def __init__(
        self,
        robot_interface: R1Interface,
        policy: BaseDiffusionPolicy,
        *,
        num_pcd_points: int,
        pcd_x_range: Tuple[float, float],
        pcd_y_range: Tuple[float, float],
        pcd_z_range: Tuple[float, float],
        pad_pcd_if_needed: bool = False,
        camera_views: Optional[List[str]] = None,
        mobile_base_vel_action_min: Tuple[float, float, float],
        mobile_base_vel_action_max: Tuple[float, float, float],
        gripper_half_width: float = 50,
        num_latest_obs: int = 2,
        num_deployed_actions: int = 8,
        device: Union[str, int, torch.device] = "cuda:0",
        policy_freq: int = 10,
        control_freq: int = 100,
        horizon_wallclock: Optional[Union[int, float]] = None,
        horizon_steps: Optional[int] = None,
        pause_mode: bool = False,
    ):
        assert not (
            horizon_wallclock is None and horizon_steps is None
        ), "Must specify either wallclock or steps horizon"
        if horizon_wallclock is not None:
            self.horizon = int(horizon_wallclock * policy_freq)
        else:
            self.horizon = horizon_steps

        self.robot_interface = robot_interface
        self.policy = policy
        self.num_pcd_points = num_pcd_points
        self._pcd_xyz_min = np.array([pcd_x_range[0], pcd_y_range[0], pcd_z_range[0]])
        self._pcd_xyz_max = np.array([pcd_x_range[1], pcd_y_range[1], pcd_z_range[1]])
        self._pad_pcd_if_needed = pad_pcd_if_needed
        self._camera_views = camera_views or []
        self._mobile_base_vel_action_min = np.array(mobile_base_vel_action_min)
        self._mobile_base_vel_action_max = np.array(mobile_base_vel_action_max)
        self.gripper_half_width = gripper_half_width
        self.num_latest_obs = num_latest_obs
        self.num_deployed_actions = num_deployed_actions
        self.device = device
        self.action_repeat = int(control_freq / policy_freq)
        self.obs_rate = rospy.Rate(policy_freq)

        self.pause_mode = pause_mode
        if pause_mode:
            urdf_path = os.path.join(ASSET_ROOT, "robot", "r1_pro", "r1_pro.urdf")
            self._viz_pb_client_id = pb.connect(pb.GUI)
            self._viz_pb_robot_id = pb.loadURDF(
                str(urdf_path),
                [0, 0, 0],
                useFixedBase=True,
                physicsClientId=self._viz_pb_client_id,
            )
            pb.resetBasePositionAndOrientation(
                self._viz_pb_robot_id,
                [0, 0, 0],
                [0, 0, 0, 1],
                physicsClientId=self._viz_pb_client_id,
            )
            self._viz_pb_robot_num_joints = pb.getNumJoints(
                self._viz_pb_robot_id, physicsClientId=self._viz_pb_client_id
            )
            for i in range(self._viz_pb_robot_num_joints):
                pb.resetJointState(
                    self._viz_pb_robot_id, i, 0, physicsClientId=self._viz_pb_client_id
                )

            self._viz_pb_robot_left_arm_joint_idxs = [[] for _ in range(6)]
            self._viz_pb_robot_right_arm_joint_idxs = [[] for _ in range(6)]
            self._viz_pb_robot_torso_joint_idxs = [[] for _ in range(4)]

            left_joint_pattern = re.compile(r"left_arm_joint[1-6]")
            right_joint_pattern = re.compile(r"right_arm_joint[1-6]")
            torso_joint_pattern = re.compile(r"torso_joint[1-4]")

            for _id in range(
                pb.getNumJoints(
                    self._viz_pb_robot_id, physicsClientId=self._viz_pb_client_id
                )
            ):
                joint_info = pb.getJointInfo(
                    self._viz_pb_robot_id, _id, physicsClientId=self._viz_pb_client_id
                )
                joint_name = joint_info[1].decode("UTF-8")
                if left_joint_pattern.match(joint_name):
                    idx = int(joint_name[-1]) - 1
                    self._viz_pb_robot_left_arm_joint_idxs[idx].append(_id)
                elif right_joint_pattern.match(joint_name):
                    idx = int(joint_name[-1]) - 1
                    self._viz_pb_robot_right_arm_joint_idxs[idx].append(_id)
                elif torso_joint_pattern.match(joint_name):
                    idx = int(joint_name[-1]) - 1
                    self._viz_pb_robot_torso_joint_idxs[idx].append(_id)
            self._viz_pb_robot_left_arm_joint_idxs = [
                idx[0] for idx in self._viz_pb_robot_left_arm_joint_idxs
            ]
            self._viz_pb_robot_right_arm_joint_idxs = [
                idx[0] for idx in self._viz_pb_robot_right_arm_joint_idxs
            ]
            self._viz_pb_robot_torso_joint_idxs = [
                idx[0] for idx in self._viz_pb_robot_torso_joint_idxs
            ]

    def rollout(self):
        obs_history = deque(maxlen=self.num_latest_obs)
        t = 0
        action_traj_pred = None
        action_idx = 0
        pbar = tqdm(total=self.horizon)
        if self.pause_mode:
            curr_qpos = self.robot_interface.last_joint_position
            left_arm_qpos, right_arm_qpos, torso_qpos = (
                curr_qpos["left_arm"],
                curr_qpos["right_arm"],
                curr_qpos["torso"],
            )
            for idx, q in zip(self._viz_pb_robot_left_arm_joint_idxs, left_arm_qpos):
                pb.resetJointState(
                    self._viz_pb_robot_id,
                    idx,
                    q,
                    physicsClientId=self._viz_pb_client_id,
                )
            for idx, q in zip(self._viz_pb_robot_right_arm_joint_idxs, right_arm_qpos):
                pb.resetJointState(
                    self._viz_pb_robot_id,
                    idx,
                    q,
                    physicsClientId=self._viz_pb_client_id,
                )
            for idx, q in zip(self._viz_pb_robot_torso_joint_idxs, torso_qpos):
                pb.resetJointState(
                    self._viz_pb_robot_id,
                    idx,
                    q,
                    physicsClientId=self._viz_pb_client_id,
                )

        while not rospy.is_shutdown() and t < self.horizon:
            if len(obs_history) == 0:
                for _ in range(self.num_latest_obs):
                    obs_history.append(self._get_normalized_obs())
            else:
                obs_history.append(self._get_normalized_obs())

            obs = any_concat(obs_history, dim=1)  # (B = 1, T = num_latest_obs, ...)

            need_inference = action_idx % self.num_deployed_actions == 0
            if need_inference:
                action_traj_pred = self.policy.act(obs)  # dict of (B = 1, T_A, ...)
                action_traj_pred = {
                    k: v[0].detach().cpu().numpy() for k, v in action_traj_pred.items()
                }  # dict of (T_A, ...)
                action_idx = 0

            action = any_slice(action_traj_pred, np.s_[action_idx])
            action = self._unnormalize_action(action)
            proceed = True
            if self.pause_mode:
                for idx, q in zip(
                    self._viz_pb_robot_right_arm_joint_idxs, action["right_arm"]
                ):
                    pb.resetJointState(
                        self._viz_pb_robot_id,
                        idx,
                        q,
                        physicsClientId=self._viz_pb_client_id,
                    )
                for idx, q in zip(
                    self._viz_pb_robot_left_arm_joint_idxs, action["left_arm"]
                ):
                    pb.resetJointState(
                        self._viz_pb_robot_id,
                        idx,
                        q,
                        physicsClientId=self._viz_pb_client_id,
                    )
                for idx, q in zip(self._viz_pb_robot_torso_joint_idxs, action["torso"]):
                    pb.resetJointState(
                        self._viz_pb_robot_id,
                        idx,
                        q,
                        physicsClientId=self._viz_pb_client_id,
                    )
                proceed = (
                    input(
                        f"""Mobile base: {action["mobile_base"]}. Proceed? (y/n):"""
                    ).lower()
                    == "y"
                )
            if proceed:
                for _ in range(self.action_repeat):
                    self.robot_interface.control(
                        arm_cmd={
                            "left": action["left_arm"],
                            "right": action["right_arm"],
                        },
                        gripper_cmd={
                            "left": action["left_gripper"],
                            "right": action["right_gripper"],
                        },
                        torso_cmd=action["torso"],
                        base_cmd=action["mobile_base"],
                    )
                if self.pause_mode:
                    self.robot_interface.stop_mobile_base()
            t += 1
            action_idx += 1
            pbar.update(1)
        self.robot_interface.stop_mobile_base()

    def _get_normalized_obs(self):
        self.obs_rate.sleep()
        pcd = self.robot_interface.last_pointcloud
        all_qpos = self.robot_interface.last_joint_position
        gripper_state = self.robot_interface.last_gripper_state
        rgbs = self.robot_interface.last_rgb

        pcd_xyz = (
            2
            * (pcd["xyz"] - self._pcd_xyz_min)
            / (self._pcd_xyz_max - self._pcd_xyz_min)
            - 1
        ).astype(np.float32)
        pcd_rgb = (pcd["rgb"] / 255).astype(np.float32)
        if pcd_xyz.shape[0] > self.num_pcd_points:
            sampling_idx = np.random.permutation(pcd_xyz.shape[0])[
                : self.num_pcd_points
            ]
            pcd_xyz = pcd_xyz[sampling_idx]
            pcd_rgb = pcd_rgb[sampling_idx]
        if self._pad_pcd_if_needed and pcd_xyz.shape[0] < self.num_pcd_points:
            N_pads = self.num_pcd_points - pcd_xyz.shape[0]
            pcd_xyz = np.concatenate(
                [pcd_xyz, np.zeros((N_pads, 3), dtype=pcd_xyz.dtype)], axis=0
            )
            pcd_rgb = np.concatenate(
                [pcd_rgb, np.zeros((N_pads, 3), dtype=pcd_rgb.dtype)], axis=0
            )
        multi_view_cameras = {}
        for view in self._camera_views:
            img = any_to_torch_tensor(
                rgbs[view]["img"], device=self.device, dtype=torch.uint8
            )
            img = rearrange(img, "H W C -> C H W")
            multi_view_cameras[f"{view}_rgb"] = img.unsqueeze(0).unsqueeze(0)

        left_arm_qpos = all_qpos["left_arm"][:6]
        left_arm_qpos = (
            2
            * (left_arm_qpos - self.left_arm_joint_low)
            / (self.left_arm_joint_high - self.left_arm_joint_low)
            - 1
        ).astype(np.float32)
        right_arm_qpos = all_qpos["right_arm"][:6]
        right_arm_qpos = (
            2
            * (right_arm_qpos - self.right_arm_joint_low)
            / (self.right_arm_joint_high - self.right_arm_joint_low)
            - 1
        ).astype(np.float32)
        torso_qpos = all_qpos["torso"]
        torso_qpos = (
            2
            * (torso_qpos - self.torso_joint_low)
            / (self.torso_joint_high - self.torso_joint_low)
            - 1
        ).astype(np.float32)

        left_gripper_position = gripper_state["left_gripper"]["gripper_position"][
            np.newaxis
        ]
        left_gripper_state = (left_gripper_position <= self.gripper_half_width).astype(
            np.float32
        ) * 2 - 1
        right_gripper_position = gripper_state["right_gripper"]["gripper_position"][
            np.newaxis
        ]
        right_gripper_state = (
            right_gripper_position <= self.gripper_half_width
        ).astype(np.float32) * 2 - 1
        odom_base_vel = self.robot_interface.curr_base_velocity
        odom_base_vel = (
            2
            * (odom_base_vel - self._mobile_base_vel_action_min)
            / (self._mobile_base_vel_action_max - self._mobile_base_vel_action_min)
            - 1
        )
        odom_base_vel = np.clip(odom_base_vel, -1, 1)

        obs_dict = {
            "pointcloud": {
                "xyz": any_to_torch_tensor(
                    pcd_xyz, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),  # (B=1, T=1, N, 3)
                "rgb": any_to_torch_tensor(
                    pcd_rgb, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),
            },
            "qpos": {
                "left_arm": any_to_torch_tensor(
                    left_arm_qpos, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),
                "right_arm": any_to_torch_tensor(
                    right_arm_qpos, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),
                "torso": any_to_torch_tensor(
                    torso_qpos, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),
                "left_gripper": any_to_torch_tensor(
                    left_gripper_state, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),
                "right_gripper": any_to_torch_tensor(
                    right_gripper_state, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),
            },
            "odom": {
                "base_velocity": any_to_torch_tensor(
                    odom_base_vel, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .unsqueeze(0),
            },
            "multi_view_cameras": multi_view_cameras,
        }
        return obs_dict

    def _unnormalize_action(
        self, action: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        mobile_base_vel_cmd = action["mobile_base"]
        mobile_base_vel_cmd = np.clip(mobile_base_vel_cmd, -1, 1)
        mobile_base_vel_cmd = (mobile_base_vel_cmd + 1) / 2 * (
            self._mobile_base_vel_action_max - self._mobile_base_vel_action_min
        ) + self._mobile_base_vel_action_min
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
        left_gripper = 1.0 if action["left_gripper"] > 0 else 0.0
        right_gripper = 1.0 if action["right_gripper"] > 0 else 0.0
        return {
            "mobile_base": mobile_base_vel_cmd,
            "left_arm": left_arm,
            "left_gripper": left_gripper,
            "right_arm": right_arm,
            "right_gripper": right_gripper,
            "torso": torso,
        }
