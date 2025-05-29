import os
import time
import argparse

import h5py
from tqdm import tqdm


def merge(args):
    data_dir = args.data_dir
    output_file = args.output_file
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    data_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".h5") or f.endswith(".hdf5")
    ]
    print(f"[INFO] Found {len(data_files)} data files in {data_dir}")

    # start merging
    merged_file = h5py.File(output_file, "w")
    # meta data
    merged_file.attrs["num_demos"] = len(data_files)
    merged_file.attrs["merging_time"] = time.strftime("%Y-%m-%d-%H-%M-%S")
    merged_file.attrs["merged_data_files"] = data_files
    # merge each demo
    for idx, fpath in tqdm(enumerate(data_files), desc="Merging data files"):
        f_to_merge = h5py.File(fpath, "r")
        demo_grp = merged_file.create_group(f"demo_{idx}")
        # passover meta data
        for key in f_to_merge.attrs:
            demo_grp.attrs[key] = f_to_merge.attrs[key]
        # obs group
        obs_grp = demo_grp.create_group("obs")
        # joint_state
        for ks in f_to_merge["obs/joint_state"]:
            for k, v in f_to_merge["obs/joint_state"][ks].items():
                obs_grp.create_dataset(f"joint_state/{ks}/{k}", data=v[:])
        # gripper_state
        for ks in f_to_merge["obs/gripper_state"]:
            for k, v in f_to_merge["obs/gripper_state"][ks].items():
                obs_grp.create_dataset(f"gripper_state/{ks}/{k}", data=v[:])
        # link_poses
        for k, v in f_to_merge["obs/link_poses"].items():
            obs_grp.create_dataset(f"link_poses/{k}", data=v[:])
        # fused point cloud
        pcd_xyz = f_to_merge["obs/point_cloud/fused/xyz"][:]  # (T, N_points, 3)
        pcd_rgb = f_to_merge["obs/point_cloud/fused/rgb"][:]
        pcd_padding_mask = f_to_merge["obs/point_cloud/fused/padding_mask"][
            :
        ]  # (T, N_points)
        obs_grp.create_dataset("point_cloud/fused/xyz", data=pcd_xyz)
        obs_grp.create_dataset("point_cloud/fused/rgb", data=pcd_rgb)
        obs_grp.create_dataset("point_cloud/fused/padding_mask", data=pcd_padding_mask)
        # odom
        odom_base_velocity = f_to_merge["obs/odom/base_velocity"][:]
        obs_grp.create_dataset("odom/base_velocity", data=odom_base_velocity)
        # multiview cameras
        head_camera_rgb = f_to_merge["obs/rgb/head/img"][:]
        head_camera_depth = f_to_merge["obs/depth/head/depth"][:]
        left_wrist_camera_rgb = f_to_merge["obs/rgb/left_wrist/img"][:]
        left_wrist_camera_depth = f_to_merge["obs/depth/left_wrist/depth"][:]
        right_wrist_camera_rgb = f_to_merge["obs/rgb/right_wrist/img"][:]
        right_wrist_camera_depth = f_to_merge["obs/depth/right_wrist/depth"][:]
        obs_grp.create_dataset("rgb/head/img", data=head_camera_rgb)
        obs_grp.create_dataset("depth/head/depth", data=head_camera_depth)
        obs_grp.create_dataset("rgb/left_wrist/img", data=left_wrist_camera_rgb)
        obs_grp.create_dataset("depth/left_wrist/depth", data=left_wrist_camera_depth)
        obs_grp.create_dataset("rgb/right_wrist/img", data=right_wrist_camera_rgb)
        obs_grp.create_dataset("depth/right_wrist/depth", data=right_wrist_camera_depth)
        # action
        action_grp = demo_grp.create_group("action")
        for k, v in f_to_merge["action"].items():
            action_grp.create_dataset(k, data=v[:])
        f_to_merge.close()
    merged_file.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing individual data files to consolidate.",
    )
    args.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="A single output file to save the consolidated data.",
    )
    args = args.parse_args()
    merge(args)
