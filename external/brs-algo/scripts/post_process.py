"""
Post-process collected raw data. Including:
    - Compute poses for left/right EEFs and head.
    - Process raw odometry data such that it is consistent with R1Interface.
"""

import os
import argparse
from math import ceil

import h5py
import numpy as np
from tqdm import tqdm
import quaternion

from brs_ctrl.kinematics import R1Kinematics


JMAP = {
    "left_wrist": "obs/joint_state/left_arm/joint_position",
    "right_wrist": "obs/joint_state/right_arm/joint_position",
    "torso": "obs/joint_state/torso/joint_position",
}
PCDKMAP = {
    "left_wrist": "obs/point_cloud/left_wrist",
    "right_wrist": "obs/point_cloud/right_wrist",
    "torso": "obs/point_cloud/head",
}
CMAP = {
    "left_wrist": "left_wrist_camera",
    "right_wrist": "right_wrist_camera",
    "torso": "head_camera",
}


def main(args):
    data_dir = args.data_dir
    chunk_size = args.process_chunk_size
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"

    kin = R1Kinematics()
    T_odom2base = kin.T_odom2base

    data_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".h5") or f.endswith(".hdf5")
    ]
    print(f"[INFO] Found {len(data_files)} data files in {data_dir}")
    for fpath in tqdm(data_files, desc="Processing data files"):
        f = h5py.File(fpath, "r+")
        all_qpos = {k: f[v][:] for k, v in JMAP.items()}  # (T, N_joints)
        raw_odom_linear_vel = f["obs/odom/linear_velocity"][:]  # (T, 3)
        raw_odom_angular_vel = f["obs/odom/angular_velocity"][:]  # (T, 3)
        # set small vel values to zero, using the same threshold as in R1Interface
        vx_zero_idxs = np.abs(raw_odom_linear_vel[:, 0]) <= 1e-2
        vy_zero_idxs = np.abs(raw_odom_linear_vel[:, 1]) <= 1e-2
        vyaw_zero_idxs = np.abs(raw_odom_angular_vel[:, 2]) <= 5e-3
        raw_odom_linear_vel[vx_zero_idxs, 0] = 0
        raw_odom_linear_vel[vy_zero_idxs, 1] = 0
        raw_odom_angular_vel[vyaw_zero_idxs, 2] = 0

        T = all_qpos[list(JMAP.keys())[0]].shape[0]
        link_poses = {
            "left_eef": {
                "position": np.zeros((T, 3), dtype=np.float32),
                "orientation": np.zeros((T, 3, 3), dtype=np.float32),
            },
            "right_eef": {
                "position": np.zeros((T, 3), dtype=np.float32),
                "orientation": np.zeros((T, 3, 3), dtype=np.float32),
            },
            "head": {
                "position": np.zeros((T, 3), dtype=np.float32),
                "orientation": np.zeros((T, 3, 3), dtype=np.float32),
            },
        }
        base_vel = np.zeros((T, 3), dtype=np.float32)  # (v_x, v_y, v_yaw)

        # process in chunks along time dimension
        N_chunks = ceil(T / chunk_size)
        for chunk_idx in tqdm(range(N_chunks), desc="Processing chunks"):
            start_t = chunk_idx * chunk_size
            end_t = min((chunk_idx + 1) * chunk_size, T)
            for t in range(start_t, end_t):
                link2base = kin.get_link_poses_in_base_link(
                    curr_left_arm_joint=all_qpos["left_wrist"][t, :6],
                    curr_right_arm_joint=all_qpos["right_wrist"][t, :6],
                    curr_torso_joint=all_qpos["torso"][t],
                )  # dict of (4, 4)
                for k in link_poses:
                    transform = link2base[k]
                    link_poses[k]["position"][t] = transform[:3, 3]
                    link_poses[k]["orientation"][t] = transform[:3, :3]

            raw_odom_linear_vel_chunk = raw_odom_linear_vel[
                start_t:end_t
            ]  # (T_chunk, 3)
            base_linear_vel_chunk = (
                T_odom2base[:3, :3] @ raw_odom_linear_vel_chunk.T
            ).T
            base_angular_vel_chunk = raw_odom_angular_vel[start_t:end_t]
            base_vel[start_t:end_t, :2] = base_linear_vel_chunk[:, :2]
            base_vel[start_t:end_t, 2] = base_angular_vel_chunk[:, 2]

        # save link poses to file
        link_pose_grp = f.create_group("obs/link_poses")
        for k in link_poses:
            rot_mat = link_poses[k]["orientation"]
            rot_quat = quaternion.as_float_array(
                quaternion.from_rotation_matrix(rot_mat)
            )  # (T, 4) in wxyz order
            # change to xyzw order since that's what pybullet uses
            rot_quat = rot_quat[..., [1, 2, 3, 0]]
            pose = np.concatenate(
                [link_poses[k]["position"], rot_quat], axis=-1
            )  # (T, 7)
            link_pose_grp.create_dataset(k, data=pose)
        # save base velocity to file
        f.create_dataset("obs/odom/base_velocity", data=base_vel)
        f.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--process_chunk_size", type=int, default=500)
    args = args.parse_args()
    main(args)
