# Wrapper around omnigibson environment allowing connection to a policy

import omnigibson as og
import json
import torch as th

class WBVIMARolloutEnv():

    OBS_MODALITIES = ["rgb", "depth_linear", "proprio"]
    SENSOR_RESOLUTION = 240

    def __init__(self, config_file_path: str, scene_file_path: str):
        
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
        self.env = og.Environment(configs=config)

    def step(self):
        """ Perform a step """


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