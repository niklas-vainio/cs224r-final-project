# Fork of og_data_replay_example to collect RGBD images and proprioception data for WB-VIMA

from omnigibson.envs import DataPlaybackWrapper
from omnigibson.envs import EnvMetric
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.utils.constants import STRUCTURE_CATEGORIES
import torch as th
import numpy as np
from omnigibson.macros import gm
import os
import omnigibson as og
import argparse
import sys
import json

from point_cloud_utils import depth_to_pcd, downsample_pcd, color_pcd_vis

# Intrinsics for external camera
# Focal length is 17mm, sensor width is 40mm, resolution is 240x240
EXTERNAL_INTRINSICS = np.array([
    [102.000,   0.000, 120.000],
    [  0.000, 102.000, 120.000],
    [  0.000,   0.000,   1.000]
])

# Intrinsics for wrist cameras
# Focal length is 17mm, sensor width is 20.995mm, resolution is 240x240
WRIST_INTRINSICS = np.array([
    [194.332,   0.000, 120.000],
    [  0.000, 194.332, 120.000],
    [  0.000,   0.000,   1.000]
])

# Simple overide class to also gather left and right EEF positions
class CustomDataPlaybackWrapper(DataPlaybackWrapper):
    
    PCD_NUM_POINTS = 4096
    
    def _process_obs(self, obs, info):
        # Custom override to process the RGBD images into a fused poitn cloud, which gets saved to observation
        
        robot = self.env.robots[0]
        base_link_pose = th.concatenate(robot.get_position_orientation()).numpy()
        
        # Left point cloud
        left_rgb = obs.pop("robot_r1::robot_r1:left_realsense_link:Camera:0::rgb")[..., :3].numpy()
        left_depth = obs.pop("robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear").numpy()
        
        left_pose = th.concatenate(robot.sensors["robot_r1:left_realsense_link:Camera:0"].get_position_orientation()).numpy()
        left_pcd = depth_to_pcd(left_depth, left_pose, base_link_pose, K=WRIST_INTRINSICS)
        left_rgb_pcd = np.concatenate([left_rgb / 255.0, left_pcd], axis=-1).reshape(-1, 6)
        
        # Right point cloud
        right_rgb = obs.pop("robot_r1::robot_r1:right_realsense_link:Camera:0::rgb")[..., :3].numpy()
        right_depth = obs.pop("robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear").numpy()
        
        right_pose = th.concatenate(robot.sensors["robot_r1:right_realsense_link:Camera:0"].get_position_orientation()).numpy()
        right_pcd = depth_to_pcd(right_depth, right_pose, base_link_pose, K=WRIST_INTRINSICS)
        right_rgb_pcd = np.concatenate([right_rgb / 255.0, right_pcd], axis=-1).reshape(-1, 6)
        
        # External camera point cloud
        external_rgb = obs.pop("external::external_sensor0::rgb")[..., :3].numpy()
        external_depth = obs.pop("external::external_sensor0::depth_linear").numpy()
        
        external_pose = th.concatenate(self.env.external_sensors["external_sensor0"].get_position_orientation()).numpy()
        external_pcd = depth_to_pcd(external_depth, external_pose, base_link_pose, K=EXTERNAL_INTRINSICS)
        external_rgb_pcd = np.concatenate([external_rgb / 255.0, external_pcd], axis=-1).reshape(-1, 6)

        # Fuse all point clouds and downsample
        fused_pcd_all = np.concatenate([left_rgb_pcd, right_rgb_pcd, external_rgb_pcd], axis=0)
        fused_pcd = downsample_pcd(fused_pcd_all, self.PCD_NUM_POINTS)
        
        obs["point_cloud_rgbxyz"] = th.from_numpy(fused_pcd) 
        
        # Remove unused cameras
        obs.pop("robot_r1::robot_r1:zed_link:Camera:0::rgb")    
        obs.pop("robot_r1::robot_r1:zed_link:Camera:0::depth_linear")    

        return obs
    

def replay_hdf5_file(hdf_input_path, write_video=False):
    """
    Replays a single HDF5 file and saves videos to a new folder
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
    """
    # Create folder with same name as HDF5 file (without extension)
    base_name = os.path.basename(hdf_input_path)
    folder_name = os.path.splitext(base_name)[0]
    folder_path = os.path.join(os.path.dirname(hdf_input_path), folder_name)
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Define output paths
    hdf_output_path = os.path.join(folder_path, f"{folder_name}_replay.hdf5")
    video_dir = folder_path

    # Move original HDF5 file to the new folder
    new_hdf_input_path = os.path.join(folder_path, base_name)
    if hdf_input_path != new_hdf_input_path:  # Avoid copying if already in target folder
        os.rename(hdf_input_path, new_hdf_input_path)
        hdf_input_path = new_hdf_input_path
    
    # Define resolution for consistency
    RESOLUTION = 240
    
    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False
    
    # Robot sensor configuration
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "depth_linear"],
            "sensor_kwargs": {
                "image_height": RESOLUTION,
                "image_width": RESOLUTION,
            },
        },
    }

    # Replace normal head camera with custom config
    external_sensors_config = []
    external_sensors_config.append({
        "sensor_type": "VisionSensor",
        "name": f"external_sensor0",
        "relative_prim_path": f"/controllable__r1pro__robot_r1/zed_link/external_sensor0",
        "modalities": ["rgb", "depth_linear"],
        "sensor_kwargs": {
            "image_height": RESOLUTION,
            "image_width": RESOLUTION,
            "horizontal_aperture": 40.0,
        },
        "position": th.tensor([0.06, 0.0, 0.01], dtype=th.float32),
        "orientation": th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
        "pose_frame": "parent",
    })

    # Create the environment
    additional_wrapper_configs = []
 
    env = CustomDataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb", "depth_linear", "proprio"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=1,
        only_successes=False,
        additional_wrapper_configs=additional_wrapper_configs,
    )
    
    # Create a list to store video writers and RGB keys
    video_writers = []
    video_keys = []

    if write_video:
        # Create video writer for robot cameras
        robot_camera_names = [
            'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb', 
            'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb', 
            'robot_r1::robot_r1:zed_link:Camera:0::rgb'
        ]
        
        for camera_name in robot_camera_names:
            video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{camera_name}.mp4"))
            video_keys.append(camera_name)
            
        
    # Playback the dataset with all video writers
    # We avoid calling playback_dataset and call playback_episode individually in order to manually
    # aggregate per-episode metrics
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        print(f" >>> Replaying episode {episode_id}")

        env.playback_episode(
            episode_id=episode_id,
            record_data=True,
            video_writers=video_writers,
            video_rgb_keys=video_keys,
        )

    # Close all video writers
    for writer in video_writers:
        writer.close()

    env.save_data()

    # Always clear the environment to free resources
    og.clear()
        
    print(f"Successfully processed {hdf_input_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    parser.add_argument("--write_video", action="store_true", help="Include this flag to write RGB and depth video files")
    
    args = parser.parse_args()
    
    if args.dir and os.path.isdir(args.dir):
        # Process all HDF5 files in the directory (non-recursively)
        hdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                    if f.lower().endswith('.hdf5') and os.path.isfile(os.path.join(args.dir, f))]
        
        if not hdf_files:
            print(f"No HDF5 files found in directory: {args.dir}")
        else:
            print(f"Found {len(hdf_files)} HDF5 files to process")
    elif args.files:
        # Process individual files specified
        hdf_files = args.files
    else:
        parser.print_help()
        print("\nError: Either --dir or --files must be specified", file=sys.stderr)
        return
    
    # Process each file
    for hdf_file in hdf_files:
        if not os.path.exists(hdf_file):
            print(f"Error: File {hdf_file} does not exist", file=sys.stderr)
            continue
            
        replay_hdf5_file(hdf_file, write_video=args.write_video)

        
    print("All done!")


if __name__ == "__main__":
    main()