from typing import List, Optional, Tuple

import os
from copy import deepcopy

import h5py
import numpy as np
from torch.utils.data import Dataset

import brs_algo.utils as U


class SeqChunkDataset(Dataset):
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    gripper_strike_low = np.array([0])
    gripper_strike_high = np.array([100])

    def __init__(
        self,
        *,
        fpath: str,
        pcd_downsample_points: int,
        pcd_x_range: Tuple[float, float],
        pcd_y_range: Tuple[float, float],
        pcd_z_range: Tuple[float, float],
        mobile_base_vel_action_min: Tuple[float, float, float],
        mobile_base_vel_action_max: Tuple[float, float, float],
        ctx_len: int,
        minimal_obs_window: int,
        load_visual_obs_in_memory: bool = True,
        multi_view_cameras: Optional[List[str]] = None,
        load_multi_view_camera_rgb: bool,
        load_multi_view_camera_depth: bool,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert os.path.exists(fpath)
        self._fpath = fpath
        self._hdf5_file = None
        self._random_state = np.random.RandomState(seed)
        self._pcd_downsample_points = pcd_downsample_points
        self._pcd_xyz_min = np.array([pcd_x_range[0], pcd_y_range[0], pcd_z_range[0]])
        self._pcd_xyz_max = np.array([pcd_x_range[1], pcd_y_range[1], pcd_z_range[1]])
        self._mobile_base_vel_action_min = np.array(mobile_base_vel_action_min)
        self._mobile_base_vel_action_max = np.array(mobile_base_vel_action_max)
        self._ctx_len = ctx_len
        self._minimal_obs_window = minimal_obs_window
        self._load_visual_obs_in_memory = load_visual_obs_in_memory
        self._multi_view_cameras = (
            multi_view_cameras if multi_view_cameras is not None else []
        )
        self._load_multi_view_camera_rgb = load_multi_view_camera_rgb
        self._load_multi_view_camera_depth = load_multi_view_camera_depth
        demo_keys = self._random_state.permutation(list(self.hdf5_file.keys()))
        self._demo_keys = demo_keys

        self._all_demos = [self._load_single_demo(demo_key) for demo_key in demo_keys]
        self._len = 0
        # compute length
        for demo in self._all_demos:
            L = U.get_batch_size(demo, strict=True)
            N_chunks = L - self._minimal_obs_window + 1
            self._len += N_chunks
        # give a random starting demo
        self._demo_ptr = np.random.randint(len(self._all_demos))
        self._demo_chunk_ptr = 0
        self._data_chunk, self._mask_chunk, self._chunk_idxs = self._chunk_demo(
            self._all_demos[self._demo_ptr]
        )
        self._demo_key = demo_keys[self._demo_ptr]

    def __getitem__(self, idx):
        # decide if we need to move to the next demo
        if self._demo_chunk_ptr >= len(self._data_chunk):
            self._demo_chunk_ptr = 0
            self._demo_ptr += 1
            if self._demo_ptr >= len(self._all_demos):
                self._demo_ptr = 0
            self._data_chunk, self._mask_chunk, self._chunk_idxs = self._chunk_demo(
                self._all_demos[self._demo_ptr]
            )
            self._demo_key = self._demo_keys[self._demo_ptr]
        data, mask = (
            self._data_chunk[self._demo_chunk_ptr],
            self._mask_chunk[self._demo_chunk_ptr],
        )
        # read visual obs from file if not loaded in memory
        if not self._load_visual_obs_in_memory:
            chunk_idx = self._chunk_idxs[self._demo_chunk_ptr]
            demo = self.hdf5_file[self._demo_key]
            # point cloud
            pcd_xyz = demo["obs/point_cloud/fused/xyz"][
                chunk_idx : chunk_idx + self._ctx_len
            ].astype(
                np.float32
            )  # (T_ctx, N, 3)
            pcd_xyz = (
                2
                * (pcd_xyz - self._pcd_xyz_min)
                / (self._pcd_xyz_max - self._pcd_xyz_min)
                - 1
            )
            pcd_rgb = (
                demo["obs/point_cloud/fused/rgb"][
                    chunk_idx : chunk_idx + self._ctx_len
                ].astype(np.uint8)
                / 255.0
            ).astype(
                np.float32
            )  # (T_ctx, N, 3)
            pcd_mask = demo["obs/point_cloud/fused/padding_mask"][
                chunk_idx : chunk_idx + self._ctx_len
            ].astype(
                bool
            )  # (T_ctx, N)
            visual_obs_dict = {
                "pointcloud": {
                    "xyz": pcd_xyz.astype(np.float32),
                    "rgb": pcd_rgb.astype(np.float32),
                    "mask": pcd_mask.astype(bool),
                }
            }
            multi_view_cameras = {}
            for camera in self._multi_view_cameras:
                if self._load_multi_view_camera_rgb:
                    # not normalize at this time because it happens in the model forward pass
                    rgb_img = demo[f"obs/rgb/{camera}/img"][
                        chunk_idx : chunk_idx + self._ctx_len
                    ].astype(
                        np.uint8
                    )  # (T_ctx, H, W, 3)
                    rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))  # (T_ctx, 3, H, W)
                    multi_view_cameras[f"{camera}_rgb"] = rgb_img
                if self._load_multi_view_camera_depth:
                    depth_img = demo[f"obs/depth/{camera}/depth"][
                        chunk_idx : chunk_idx + self._ctx_len
                    ].astype(
                        np.float32
                    )  # (T_ctx, H, W)
                    multi_view_cameras[f"{camera}_depth"] = depth_img
            visual_obs_dict["multi_view_cameras"] = multi_view_cameras
            # pad data chunks to equal length of ctx_len
            data_structure = deepcopy(
                U.any_slice(visual_obs_dict, np.s_[0:1])
            )  # (T = 1, ...)
            padded_visual_obs_dict = U.any_concat(
                [
                    visual_obs_dict,
                ]
                + [U.any_ones_like(data_structure)]
                * (self._ctx_len - U.get_batch_size(visual_obs_dict)),
                dim=0,
            )
            data.update(padded_visual_obs_dict)
        self._demo_chunk_ptr += 1

        # downsample point cloud if needed
        raw_pcd = data["pointcloud"]
        raw_pcd_xyz, raw_pcd_rgb, raw_pcd_pad_mask = (
            raw_pcd["xyz"],
            raw_pcd["rgb"],
            raw_pcd["mask"],
        )
        downsampled_xyz, downsampled_rgb = [], []
        for xyz, rgb, pad_mask in zip(raw_pcd_xyz, raw_pcd_rgb, raw_pcd_pad_mask):
            xyz = xyz[pad_mask]
            rgb = rgb[pad_mask]
            N_points = xyz.shape[0]
            if N_points > self._pcd_downsample_points:
                sampling_idx = self._random_state.permutation(N_points)[
                    : self._pcd_downsample_points
                ]
                downsampled_xyz.append(xyz[sampling_idx])
                downsampled_rgb.append(rgb[sampling_idx])
            elif N_points < self._pcd_downsample_points:
                N_pad = self._pcd_downsample_points - N_points
                padded_xyz = np.concatenate(
                    [xyz, np.zeros((N_pad, 3), dtype=xyz.dtype)], axis=0
                )
                padded_rgb = np.concatenate(
                    [rgb, np.zeros((N_pad, 3), dtype=rgb.dtype)], axis=0
                )
                downsampled_xyz.append(padded_xyz)
                downsampled_rgb.append(padded_rgb)
            else:
                downsampled_xyz.append(xyz)
                downsampled_rgb.append(rgb)
        downsampled_xyz = np.stack(downsampled_xyz, axis=0)
        downsampled_rgb = np.stack(downsampled_rgb, axis=0)
        data = {
            "pointcloud": {
                "xyz": downsampled_xyz,
                "rgb": downsampled_rgb,
            },
            "qpos": data["qpos"],
            "link_poses": data["link_poses"],
            "odom": data["odom"],
            "actions": data["actions"],
            "pad_mask": mask,
            "multi_view_cameras": data["multi_view_cameras"],
        }
        return data

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self._fpath, "r", swmr=True, libver="latest")
        return self._hdf5_file

    def _load_single_demo(self, demo_key):
        demo = self.hdf5_file[demo_key]

        # process actions
        mobile_base_vel_action = demo["action/mobile_base"][:].astype(
            np.float32
        )  # (T, 3)
        mobile_base_vel_action = (
            2
            * (mobile_base_vel_action - self._mobile_base_vel_action_min)
            / (self._mobile_base_vel_action_max - self._mobile_base_vel_action_min)
            - 1
        )
        mobile_base_vel_action = np.clip(mobile_base_vel_action, -1, 1)
        left_arm_action = demo["action/left_arm"][:].astype(np.float32)  # (T, 6)
        left_arm_action = (
            2
            * (left_arm_action - self.left_arm_joint_low)
            / (self.left_arm_joint_high - self.left_arm_joint_low)
            - 1
        )
        right_arm_action = demo["action/right_arm"][:].astype(np.float32)  # (T, 6)
        right_arm_action = (
            2
            * (right_arm_action - self.right_arm_joint_low)
            / (self.right_arm_joint_high - self.right_arm_joint_low)
            - 1
        )
        torso_action = demo["action/torso"][:].astype(np.float32)  # (T, 4)
        torso_action = (
            2
            * (torso_action - self.torso_joint_low)
            / (self.torso_joint_high - self.torso_joint_low)
            - 1
        )
        # because action/left/right_gripper is gripper stroke (in cm), convert to binary (1 for close, 0 for open)
        left_gripper_action = (demo["action/left_gripper"][:] <= 50).astype(np.float32)[
            ..., np.newaxis
        ]  # (T, 1)
        left_gripper_action = 2 * left_gripper_action - 1
        right_gripper_action = (demo["action/right_gripper"][:] <= 50).astype(
            np.float32
        )[
            ..., np.newaxis
        ]  # (T, 1)
        right_gripper_action = 2 * right_gripper_action - 1
        action_dict = {
            "mobile_base": mobile_base_vel_action.astype(np.float32),
            "left_arm": left_arm_action.astype(np.float32),
            "left_gripper": left_gripper_action.astype(np.float32),
            "right_arm": right_arm_action.astype(np.float32),
            "right_gripper": right_gripper_action.astype(np.float32),
            "torso": torso_action.astype(np.float32),
        }

        # process observations
        # qpos
        left_arm_qpos = demo["obs/joint_state/left_arm/joint_position"][:, :6].astype(
            np.float32
        )  # (T, 6)
        left_arm_qpos = (
            2
            * (left_arm_qpos - self.left_arm_joint_low)
            / (self.left_arm_joint_high - self.left_arm_joint_low)
            - 1
        )
        right_arm_qpos = demo["obs/joint_state/right_arm/joint_position"][:, :6].astype(
            np.float32
        )  # (T, 6)
        right_arm_qpos = (
            2
            * (right_arm_qpos - self.right_arm_joint_low)
            / (self.right_arm_joint_high - self.right_arm_joint_low)
            - 1
        )
        torso_qpos = demo["obs/joint_state/torso/joint_position"][:].astype(
            np.float32
        )  # (T, 4)
        torso_qpos = (
            2
            * (torso_qpos - self.torso_joint_low)
            / (self.torso_joint_high - self.torso_joint_low)
            - 1
        )
        left_gripper_qpos = demo["obs/gripper_state/left_gripper/gripper_position"][
            :
        ].astype(np.float32)[..., np.newaxis]
        # rectify gripper state to +1 for close, -1 for open
        left_gripper_qpos = (left_gripper_qpos <= 50).astype(np.float32) * 2 - 1
        right_gripper_qpos = demo["obs/gripper_state/right_gripper/gripper_position"][
            :
        ].astype(np.float32)[..., np.newaxis]
        right_gripper_qpos = (right_gripper_qpos <= 50).astype(np.float32) * 2 - 1
        # odometry base velocity
        odom_base_vel = demo["obs/odom/base_velocity"][:].astype(np.float32)  # (T, 3)
        odom_base_vel = (
            2
            * (odom_base_vel - self._mobile_base_vel_action_min)
            / (self._mobile_base_vel_action_max - self._mobile_base_vel_action_min)
            - 1
        )
        odom_base_vel = np.clip(odom_base_vel, -1, 1)

        obs_dict = {
            "qpos": {
                "torso": torso_qpos.astype(np.float32),  # (T, 4)
                "left_arm": left_arm_qpos.astype(np.float32),  # (T, 6)
                "right_arm": right_arm_qpos.astype(np.float32),  # (T, 6)
                "left_gripper": left_gripper_qpos.astype(np.float32),  # (T, 1)
                "right_gripper": right_gripper_qpos.astype(np.float32),  # (T, 1)
            },
            "link_poses": {
                "left_eef": demo["obs/link_poses/left_eef"][:].astype(
                    np.float32
                ),  # (T, 7)
                "right_eef": demo["obs/link_poses/right_eef"][:].astype(
                    np.float32
                ),  # (T, 7)
                "head": demo["obs/link_poses/head"][:].astype(np.float32),  # (T, 7)
            },
            "odom": {
                "base_velocity": odom_base_vel.astype(np.float32),  # (T, 3)
            },
        }
        if self._load_visual_obs_in_memory:
            # point cloud
            pcd_xyz = demo["obs/point_cloud/fused/xyz"][:].astype(
                np.float32
            )  # (T, N, 3)
            pcd_xyz = (
                2
                * (pcd_xyz - self._pcd_xyz_min)
                / (self._pcd_xyz_max - self._pcd_xyz_min)
                - 1
            )
            visual_obs_dict = {
                "pointcloud": {
                    "xyz": pcd_xyz.astype(np.float32),  # (T, N, 3)
                    "rgb": (
                        demo["obs/point_cloud/fused/rgb"][:].astype(np.uint8) / 255.0
                    ).astype(
                        np.float32
                    ),  # (T, N, 3)
                    "mask": demo["obs/point_cloud/fused/padding_mask"][:].astype(
                        bool
                    ),  # (T, N)
                },
            }
            multi_view_cameras = {}
            for camera in self._multi_view_cameras:
                if self._load_multi_view_camera_rgb:
                    # not normalize at this time because it happens in the model forward pass
                    rgb_img = demo[f"obs/rgb/{camera}/img"][:].astype(
                        np.uint8
                    )  # (T, H, W, 3)
                    rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))  # (T, 3, H, W)
                    multi_view_cameras[f"{camera}_rgb"] = rgb_img
                if self._load_multi_view_camera_depth:
                    depth_img = demo[f"obs/depth/{camera}/depth"][:].astype(
                        np.float32
                    )  # (T, H, W)
                    multi_view_cameras[f"{camera}_depth"] = depth_img
            visual_obs_dict["multi_view_cameras"] = multi_view_cameras
            obs_dict.update(visual_obs_dict)

        assert U.get_batch_size(action_dict, strict=True) == U.get_batch_size(
            obs_dict, strict=True
        )
        rtn = {"actions": action_dict}
        rtn.update(obs_dict)
        return rtn

    def _chunk_demo(self, demo):
        data_chunks = []
        chunk_idxs = []
        L = U.get_batch_size(demo, strict=True)
        assert L >= self._minimal_obs_window >= 1
        N_chunks = L - self._minimal_obs_window + 1
        # split into chunks
        for chunk_idx in range(N_chunks):
            s = np.s_[chunk_idx : chunk_idx + self._ctx_len]
            chunk_idxs.append(chunk_idx)
            data_chunks.append(U.any_slice(demo, s))
        # pad data chunks to equal length of ctx_len
        data_structure = deepcopy(
            U.any_slice(data_chunks[0], np.s_[0:1])
        )  # (T = 1, ...)
        padded_data_chunks = [
            U.any_concat(
                [
                    _chunk,
                ]
                + [U.any_ones_like(data_structure)]
                * (self._ctx_len - U.get_batch_size(_chunk)),
                dim=0,
            )
            for _chunk in data_chunks
        ]  # list of (ctx_len, ...)
        mask_chunks = [
            U.any_concat(
                [
                    np.ones((U.get_batch_size(_chunk),), dtype=bool),
                    np.zeros((self._ctx_len - U.get_batch_size(_chunk),), dtype=bool),
                ],
                dim=0,
            )
            for _chunk in data_chunks
        ]  # list of (ctx_len,)
        return padded_data_chunks, mask_chunks, chunk_idxs

    def __len__(self):
        return self._len


class ActionSeqChunkDataset(SeqChunkDataset):
    def __init__(
        self,
        *,
        fpath: str,
        pcd_downsample_points: int,
        pcd_x_range: Tuple[float, float],
        pcd_y_range: Tuple[float, float],
        pcd_z_range: Tuple[float, float],
        mobile_base_vel_action_min: Tuple[float, float, float],
        mobile_base_vel_action_max: Tuple[float, float, float],
        load_visual_obs_in_memory: bool = True,
        multi_view_cameras: Optional[List[str]] = None,
        load_multi_view_camera_rgb: bool,
        load_multi_view_camera_depth: bool,
        seed: Optional[int] = None,
        action_prediction_horizon: int,
        obs_window_size: int,
    ):
        self._action_prediction_horizon = action_prediction_horizon
        super().__init__(
            fpath=fpath,
            pcd_downsample_points=pcd_downsample_points,
            pcd_x_range=pcd_x_range,
            pcd_y_range=pcd_y_range,
            pcd_z_range=pcd_z_range,
            mobile_base_vel_action_min=mobile_base_vel_action_min,
            mobile_base_vel_action_max=mobile_base_vel_action_max,
            seed=seed,
            multi_view_cameras=multi_view_cameras,
            load_multi_view_camera_rgb=load_multi_view_camera_rgb,
            load_multi_view_camera_depth=load_multi_view_camera_depth,
            minimal_obs_window=obs_window_size,
            load_visual_obs_in_memory=load_visual_obs_in_memory,
            ctx_len=obs_window_size,
        )

    def _load_single_demo(self, demo_key):
        demo = super()._load_single_demo(demo_key)
        action_dict = demo["actions"]

        # make actions from (T, A) to (T, L_pred_horizon, A)
        # need to construct a mask
        action_chunks = []
        action_chunk_masks = []
        action_structure = deepcopy(U.any_slice(action_dict, np.s_[0:1]))  # (1, A)
        for t in range(U.get_batch_size(action_dict, strict=True)):
            action_chunk = U.any_slice(
                action_dict, np.s_[t : t + self._action_prediction_horizon]
            )
            action_chunk_size = U.get_batch_size(action_chunk, strict=True)
            pad_size = self._action_prediction_horizon - action_chunk_size
            mask = U.any_concat(
                [
                    np.ones((action_chunk_size,), dtype=bool),
                    np.zeros((pad_size,), dtype=bool),
                ],
                dim=0,
            )  # (L_pred_horizon,)
            action_chunk = U.any_concat(
                [
                    action_chunk,
                ]
                + [U.any_ones_like(action_structure)] * pad_size,
                dim=0,
            )  # (L_pred_horizon, A)
            action_chunks.append(action_chunk)
            action_chunk_masks.append(mask)
        action_chunks = U.any_stack(action_chunks, dim=0)  # (T, L_pred_horizon, A)
        action_chunk_masks = np.stack(action_chunk_masks, axis=0)  # (T, L_pred_horizon)

        demo["action_chunks"] = action_chunks
        demo["action_chunk_masks"] = action_chunk_masks
        return demo

    def __getitem__(self, idx):
        # decide if we need to move to the next demo
        if self._demo_chunk_ptr >= len(self._data_chunk):
            self._demo_chunk_ptr = 0
            self._demo_ptr += 1
            if self._demo_ptr >= len(self._all_demos):
                self._demo_ptr = 0
            self._data_chunk, self._mask_chunk, self._chunk_idxs = self._chunk_demo(
                self._all_demos[self._demo_ptr]
            )
            self._demo_key = self._demo_keys[self._demo_ptr]
        data, mask = (
            self._data_chunk[self._demo_chunk_ptr],
            self._mask_chunk[self._demo_chunk_ptr],
        )
        # read visual obs from file if not loaded in memory
        if not self._load_visual_obs_in_memory:
            chunk_idx = self._chunk_idxs[self._demo_chunk_ptr]
            demo = self.hdf5_file[self._demo_key]
            # point cloud
            pcd_xyz = demo["obs/point_cloud/fused/xyz"][
                chunk_idx : chunk_idx + self._ctx_len
            ].astype(
                np.float32
            )  # (T_ctx, N, 3)
            pcd_xyz = (
                2
                * (pcd_xyz - self._pcd_xyz_min)
                / (self._pcd_xyz_max - self._pcd_xyz_min)
                - 1
            )
            pcd_rgb = (
                demo["obs/point_cloud/fused/rgb"][
                    chunk_idx : chunk_idx + self._ctx_len
                ].astype(np.uint8)
                / 255.0
            ).astype(
                np.float32
            )  # (T_ctx, N, 3)
            pcd_mask = demo["obs/point_cloud/fused/padding_mask"][
                chunk_idx : chunk_idx + self._ctx_len
            ].astype(
                bool
            )  # (T_ctx, N)
            visual_obs_dict = {
                "pointcloud": {
                    "xyz": pcd_xyz.astype(np.float32),
                    "rgb": pcd_rgb.astype(np.float32),
                    "mask": pcd_mask.astype(bool),
                }
            }
            multi_view_cameras = {}
            for camera in self._multi_view_cameras:
                if self._load_multi_view_camera_rgb:
                    # not normalize at this time because it happens in the model forward pass
                    rgb_img = demo[f"obs/rgb/{camera}/img"][
                        chunk_idx : chunk_idx + self._ctx_len
                    ].astype(
                        np.uint8
                    )  # (T_ctx, H, W, 3)
                    rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))  # (T_ctx, 3, H, W)
                    multi_view_cameras[f"{camera}_rgb"] = rgb_img
                if self._load_multi_view_camera_depth:
                    depth_img = demo[f"obs/depth/{camera}/depth"][
                        chunk_idx : chunk_idx + self._ctx_len
                    ].astype(
                        np.float32
                    )  # (T_ctx, H, W)
                    multi_view_cameras[f"{camera}_depth"] = depth_img
            visual_obs_dict["multi_view_cameras"] = multi_view_cameras
            # pad data chunks to equal length of ctx_len
            data_structure = deepcopy(
                U.any_slice(visual_obs_dict, np.s_[0:1])
            )  # (T = 1, ...)
            padded_visual_obs_dict = U.any_concat(
                [
                    visual_obs_dict,
                ]
                + [U.any_ones_like(data_structure)]
                * (self._ctx_len - U.get_batch_size(visual_obs_dict)),
                dim=0,
            )
            data.update(padded_visual_obs_dict)
        self._demo_chunk_ptr += 1

        # downsample point cloud if needed
        raw_pcd = data["pointcloud"]
        raw_pcd_xyz, raw_pcd_rgb, raw_pcd_pad_mask = (
            raw_pcd["xyz"],
            raw_pcd["rgb"],
            raw_pcd["mask"],
        )
        downsampled_xyz, downsampled_rgb = [], []
        for xyz, rgb, pad_mask in zip(raw_pcd_xyz, raw_pcd_rgb, raw_pcd_pad_mask):
            xyz = xyz[pad_mask]
            rgb = rgb[pad_mask]
            N_points = xyz.shape[0]
            if N_points > self._pcd_downsample_points:
                sampling_idx = self._random_state.permutation(N_points)[
                    : self._pcd_downsample_points
                ]
                downsampled_xyz.append(xyz[sampling_idx])
                downsampled_rgb.append(rgb[sampling_idx])
            elif N_points < self._pcd_downsample_points:
                N_pad = self._pcd_downsample_points - N_points
                padded_xyz = np.concatenate(
                    [xyz, np.zeros((N_pad, 3), dtype=xyz.dtype)], axis=0
                )
                padded_rgb = np.concatenate(
                    [rgb, np.zeros((N_pad, 3), dtype=rgb.dtype)], axis=0
                )
                downsampled_xyz.append(padded_xyz)
                downsampled_rgb.append(padded_rgb)
            else:
                downsampled_xyz.append(xyz)
                downsampled_rgb.append(rgb)
        downsampled_xyz = np.stack(downsampled_xyz, axis=0)
        downsampled_rgb = np.stack(downsampled_rgb, axis=0)
        data = {
            "pointcloud": {
                "xyz": downsampled_xyz,
                "rgb": downsampled_rgb,
            },
            "qpos": data["qpos"],
            "odom": data["odom"],
            "link_poses": data["link_poses"],
            "action_chunks": data["action_chunks"],
            "pad_mask": data["action_chunk_masks"] & mask[:, None],
            "multi_view_cameras": data["multi_view_cameras"],
        }
        return data
