# Helper script for exporting a point cloud from an hdf5 file containing RGB + Depth images
# Niklas Vainio
# 5/20/25

# (copied here frm data_processing - should porbably set up an overall package to allow easier relative imports)

import argparse
import h5py
import os
import open3d as o3d
import fpsample
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R
import omnigibson.utils.transform_utils as T



def color_pcd_vis(color_pcd):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:,3:]) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', color_pcd.shape[0])

def depth_to_pcd(
        depth,
        pose,
        base_link_pose,
        K,
        max_depth=20,
    ):

    # get the homogeneous transformation matrix from quaternion
    pos = pose[:3]
    quat = pose[3:]
    rot = R.from_quat(quat)  # scipy expects [x, y, z, w]
    rot_add = R.from_euler('x', np.pi).as_matrix() # handle the cam_to_img transformation
    rot_matrix = rot.as_matrix() @ rot_add   # 3x3 rotation matrix
    world_to_cam_tf = np.eye(4)
    world_to_cam_tf[:3, :3] = rot_matrix
    world_to_cam_tf[:3, 3] = pos

    # filter depth
    mask = depth > max_depth
    depth[mask] = 0
    h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij", sparse=False)
    assert depth.min() >= 0
    u = x
    v = y
    uv = np.dstack((u, v, np.ones_like(u))) # (img_width, img_height, 3)

    Kinv = np.linalg.inv(K)
    
    pc = depth.reshape(-1, 1) * (uv.reshape(-1, 3) @ Kinv.T)
    pc = pc.reshape(h, w, 3)
    pc = np.concatenate([pc.reshape(-1, 3), np.ones((h * w, 1))], axis=-1)  # shape (H*W, 4)

    world_to_robot_tf = T.pose2mat((th.from_numpy(base_link_pose[:3]), th.from_numpy(base_link_pose[3:]))).numpy()
    robot_to_world_tf = np.linalg.inv(world_to_robot_tf)
    pc = (pc @ world_to_cam_tf.T @ robot_to_world_tf.T)[:, :3].reshape(h, w, 3)

    return pc

def downsample_pcd(color_pcd, num_points):
    if color_pcd.shape[0] > num_points:
        pc = color_pcd[:, 3:]
        color_img = color_pcd[:, :3]
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, num_points, h=5)
        pc = pc[kdline_fps_samples_idx]
        color_img = color_img[kdline_fps_samples_idx]
        color_pcd = np.concatenate([color_img, pc], axis=-1)
    else:
        # randomly sample points
        pad_number_of_points = num_points - color_pcd.shape[0]
        random_idx = np.random.choice(color_pcd.shape[0], pad_number_of_points, replace=True)
        pad_pcd = color_pcd[random_idx]
        color_pcd = np.concatenate([color_pcd, pad_pcd], axis=0)
        # raise ValueError("color_pcd shape is smaller than num_points_to_sample")
    return color_pcd