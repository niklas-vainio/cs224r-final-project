# Script for evaluating the policy in the environment
# Niklas Vainio
# 06/03/2025

import tyro
from wbvima_rollout_env import WBVIMARolloutEnv

@dataclass
class Args:
    task_name: str = "picking_up_trash"
    max_ep_len: int = 5000
    # policy_path: str
    config_path: str = "env_config.json"
    scene_path: str = "scene_config.json"

def main(args: Args):
    # Launch OmniGibson environment
    env = WBVIMARolloutEnv(
        args.config_path,
        args.scene_path
    )

    breakpoint()

    # done = False
    # i = 0

    # # Get initial observation

    # while i < args.max_ep_len and not done: 
    #     # Get policy action
    #     # Step environment

    #     i += 1

    # # Output data
    # pass


if __name__ == "__main__":
    main(tyro.cli(Args))