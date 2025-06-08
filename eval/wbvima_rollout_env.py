# Wrapper around omnigibson environment allowing connection to a policy

import omnigibson as og
import omnigibson.macros as gm
import json
import torch as th
import numpy as np

from wbvima_policy_wrapper import WBVIMAPolicyWrapper
from point_cloud_utils import depth_to_pcd, downsample_pcd, color_pcd_vis

class WBVIMARolloutEnv():

    OBS_MODALITIES = ["rgb", "depth_linear", "proprio"]
    SENSOR_RESOLUTION = 240
    PCD_NUM_POINTS = 4096

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

    def __init__(self, 
                 max_ep_len: int,
                 config_file_path: str, 
                 scene_file_path: str, 
                 policy_checkpoint_path: str
                 ):
        
        # Set og settings for maximum speed
        gm.ENABLE_FLATCACHE = True
        
        # Maintain running data dict
        self.log = {"data": []}

        self.max_ep_len = max_ep_len

        # Initialize policy
        print(">>> Loading policy...")
        self.policy_wrapper = WBVIMAPolicyWrapper(policy_checkpoint_path)
        
        # Initialize configuration dict
        with open(config_file_path, "r") as config_file:
            config = json.loads(config_file.read())

        robot_sensor_config, external_sensor_configs = self._get_sensor_configs()

        # Taken from data_wrapper.py in omnigibson env
        config["env"]["action_frequency"] = 1000.0
        config["env"]["rendering_frequency"] = 1000.0
        config["env"]["physics_frequency"] = 1000.0
        config["env"]["flatten_obs_space"] = True

        for robot_cfg in config["robots"]:
            robot_cfg["obs_modalities"] = self.OBS_MODALITIES
            robot_cfg["sensor_config"] = robot_sensor_config

        config["env"]["external_sensors"] = external_sensor_configs

        with open(scene_file_path, "r") as scene_file:
            scene_config = json.loads(scene_file.read())

        config["scene"]["scene_file"] = scene_config
        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False

        # Initialize og env
        print(">>> Loading environment...")
        self.env = og.Environment(configs=config)

    def rollout_episode(self, reset_env=False, log_file=None):
        """ Perform a full rollout in the environment """
        if reset_env:
            print(">>> Resetting environment...")
            self.env.reset()


        # Get initial observation
        print(">>> Starting rollout!")
        obs_raw, info = self.env.get_obs()
        obs = self._process_obs(obs_raw, info)

        total_reward = 0

        for i in range(self.max_ep_len):
            # Query policy
            action = self.policy_wrapper.query_action(obs)
            print(action)
            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            obs = self._process_obs(next_obs_raw, info)

            total_reward += reward

            # Add to log
            log_dict = {"step": i, "reward": reward, "terminated": terminated, "truncated": truncated}
            print(log_dict)
            self.log["data"].append(log_dict)

            # Exit if terminated or truncated
            if terminated:
                print("   !!! Episode terminated: exiting")
                break
            
            if truncated:
                print("   !!! Episode truncated: exiting!")
                break
        
        if log_file:
            self._save_log_to_file(log_file)

        print(f">>> Rollout complete!       # steps = {i+1}, total reward = {total_reward:.3f}")
            

    def _save_log_to_file(self, output_file: str):
        """
        Save internal episode log to file
        """
        with open(output_file, "w+") as file:
            file.write(json.dumps(self.log, indent=4))
        print(f">>> Saved log state to {output_file}!")


    def _get_sensor_configs(self) -> tuple[dict, dict]:
        """
        Return robot and external sensor configs
        """
        # Robot sensor configuration
        robot_sensor_config = {
            "VisionSensor": {
                "modalities": ["rgb", "depth_linear"],
                "sensor_kwargs": {
                    "image_height": self.SENSOR_RESOLUTION,
                    "image_width": self.SENSOR_RESOLUTION,
                }
            }
        }

        external_sensor_configs = [
            {
                "sensor_type": "VisionSensor",
                "name": f"external_sensor0",
                "relative_prim_path": f"/controllable__r1pro__robot_r1/zed_link/external_sensor0",
                "modalities": ["rgb", "depth_linear"],
                "sensor_kwargs": {
                    "image_height": self.SENSOR_RESOLUTION,
                    "image_width": self.SENSOR_RESOLUTION,
                    "horizontal_aperture": 40.0,
                },
                "position": th.tensor([0.06, 0.0, 0.01], dtype=th.float32),
                "orientation": th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
                "pose_frame": "parent",
            }
        ]

        return robot_sensor_config, external_sensor_configs


    def _process_obs(self, obs_raw, info):
        # Custom override to process the RGBD images into a fused poitn cloud, which gets saved to observation
        # Copied from data_processing (should probably not be repeating this)
        robot = self.env.robots[0]
        base_link_pose = th.concatenate(robot.get_position_orientation()).numpy()
        
        # Left point cloud
        left_rgb = obs_raw.pop("robot_r1::robot_r1:left_realsense_link:Camera:0::rgb")[..., :3].numpy()
        left_depth = obs_raw.pop("robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear").numpy()
        
        left_pose = th.concatenate(robot.sensors["robot_r1:left_realsense_link:Camera:0"].get_position_orientation()).numpy()
        left_pcd = depth_to_pcd(left_depth, left_pose, base_link_pose, K=self.WRIST_INTRINSICS)
        left_rgb_pcd = np.concatenate([left_rgb / 255.0, left_pcd], axis=-1).reshape(-1, 6)
        
        # Right point cloud
        right_rgb = obs_raw.pop("robot_r1::robot_r1:right_realsense_link:Camera:0::rgb")[..., :3].numpy()
        right_depth = obs_raw.pop("robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear").numpy()
        
        right_pose = th.concatenate(robot.sensors["robot_r1:right_realsense_link:Camera:0"].get_position_orientation()).numpy()
        right_pcd = depth_to_pcd(right_depth, right_pose, base_link_pose, K=self.WRIST_INTRINSICS)
        right_rgb_pcd = np.concatenate([right_rgb / 255.0, right_pcd], axis=-1).reshape(-1, 6)
        
        # External camera point cloud
        external_rgb = obs_raw.pop("external::external_sensor0::rgb")[..., :3].numpy()
        external_depth = obs_raw.pop("external::external_sensor0::depth_linear").numpy()
        
        external_pose = th.concatenate(self.env.external_sensors["external_sensor0"].get_position_orientation()).numpy()
        external_pcd = depth_to_pcd(external_depth, external_pose, base_link_pose, K=self.EXTERNAL_INTRINSICS)
        external_rgb_pcd = np.concatenate([external_rgb / 255.0, external_pcd], axis=-1).reshape(-1, 6)

        # Fuse all point clouds and downsample
        fused_pcd_all = np.concatenate([left_rgb_pcd, right_rgb_pcd, external_rgb_pcd], axis=0)
        fused_pcd = downsample_pcd(fused_pcd_all, self.PCD_NUM_POINTS)

        # Format in BRS compatible way
        obs = {
            "proprio": {
                "all": obs_raw["robot_r1::proprio"].numpy()
            },
            "pointcloud": {
                "xyz": fused_pcd[:, 3:].astype(np.float32),
                "rgb": fused_pcd[:, :3].astype(np.float32)
            }
        }
        return obs
