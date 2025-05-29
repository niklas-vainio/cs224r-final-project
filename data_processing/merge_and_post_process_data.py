# Script to post process all exported obsevration data into BRS-compatible format

import os
import argparse
import h5py
import numpy as np


def get_num_demos(file: h5py.File) -> int:
    """ Return the number of episodes in the specified hdf5 file """
    return file["data"].attrs["n_episodes"]


def get_episode_lengths(file: h5py.File) -> list[int]:
    """ Return the number of steps in each episode """
    episode_lengths = []

    for name, node in file["data"].items():
        num_steps = node["action"].shape[0]
        episode_lengths.append(num_steps)

    return episode_lengths


def covert_and_wrtie_demo(demo: h5py.Group, output_hf: h5py.File, global_count: int):
    """ 
    Convert the dataset for a demo into a BRS-compatible dictionary and write
    it to the given output file
    """
    data = {
        "action": {},
        "obs": {
            "proprio": {},
            "point_cloud": {
                "fused": {}
            },

        },
    }

    # Parse action
    actions = demo["action"].astype(np.float32) 
    
    data["action"]["mobile_base"] = actions[:, 0:3]     # 0-2: base (3 DOF)
    data["action"]["left_arm"] = actions[:, 7:14]       # 7-13: left arm (7 DOF)
    data["action"]["left_gripper"] = actions[:, 14]     # 14: left gripper (1 DOF)
    data["action"]["right_arm"] = actions[:, 15:22]     # 15-21: right arm (7 DOF)
    data["action"]["right_gripper"] = actions[:, 22]    # 22: right grippre (1 DOF)
    data["action"]["torso"] = actions[:, 3:7]           # 3-6: torso (4 DOF)

    # Parse observation (proprioception and point cloud)
    proptio_obs = demo["obs/robot_r1::proprio"].astype(np.float32)
    data["obs"]["proprio"]["all"] = proptio_obs 

    point_cloud_observation = demo["obs/point_cloud_rgbxyz"].astype(np.float32)
    num_steps = point_cloud_observation.shape[0]
    num_points = point_cloud_observation.shape[1]

    data["obs"]["point_cloud"]["fused"]["rgb"] = point_cloud_observation[:, :, :3]
    data["obs"]["point_cloud"]["fused"]["xyz"] = point_cloud_observation[:, :, 3:]
    data["obs"]["point_cloud"]["fused"]["padding_mask"] = np.ones((num_steps, num_points), dtype=bool) # No padding: all true

    # Write to file
    def write_dict_to_hdf5(h5file: h5py.File, path: str, d: dict):
        # Recusrively write nested dict
        for key, value in d.items():
            if isinstance(value, dict):
                grp = h5file.require_group(f"{path}/{key}")
                write_dict_to_hdf5(h5file, f"{path}/{key}", value)
            else:
                h5file.create_dataset(f"{path}/{key}", data=value)

    write_dict_to_hdf5(output_hf, f"demo_{global_count}", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_data.hdf5")

    args = parser.parse_args()

    # Recursively find all replay hdf5 files
    matches = []
    for dirpath, dirnames, filenames in os.walk(args.dir):
        for filename in filenames:
            if filename.endswith("_replay.hdf5"):
                matches.append(os.path.join(dirpath, filename))

    # Sort alphabetically
    matches.sort(key=lambda x: os.path.basename(x))

    if len(matches) == 0:
        print("No replay hdf5 files found in this folder!")
        exit(1)


    print("Exporting the following files:")
    for match in matches:
        print(f" * {match}")

    # Open output hdf5 file
    with h5py.File(args.output, "w") as output_hf:
        
        demo_count = 0

        # Replay all demo files
        for file_path in matches:
            with h5py.File(file_path, "r") as hf:
                for i in range(get_num_demos(hf)):

                    # Remove demos which are very long or very short
                    demo_group = hf[f"data/demo_{i}"]
                    demo_length = demo_group["action"].shape[0]

                    if 2000 <= demo_length <= 5000:
                        covert_and_wrtie_demo(hf[f"data/demo_{i}"], output_hf, demo_count)
                        print(f"  >> Saved demo {demo_count} from {file_path}")
                        demo_count += 1
                    else:
                        print(f"    >> Skipping demo due to invalid length {demo_length}")

            print("")

    print("")
    print(f"Saved all data to {args.output}!")
