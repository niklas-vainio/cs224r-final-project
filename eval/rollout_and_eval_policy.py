# Script for evaluating the policy in the environment
# Niklas Vainio
# 06/03/2025

import tyro
from wbvima_rollout_env import WBVIMARolloutEnv

@dataclass
class Args:
    task_name: str = "picking_up_trash"
    max_ep_len: int = 5000
    num_episodes: int = 1
    config_path: str = "env_config.json"
    scene_path: str = "scene_config.json"
    policy_checkpoint: str


def main(args: Args):
    # Launch OmniGibson environment and run rollout
    env = WBVIMARolloutEnv(
        max_ep_len=args.max_ep_len,
        config_file_path=args.config_path,
        scene_file_path=args.scene_path,
        policy_checkpoint_path=args.policy_checkpoint
    )

    # Run rollouts
    for i in range(args.num_episodes):
        print(f"[main] Beginning rollout number {i+1}/{args.num_episodes}!")
        env.rollout_episode(
            reset_env=(i != 0),
            log_file=f"eval_log_ep{i}.json",
        )
        print("\n\n\n\n\n")

    print("All done!")


if __name__ == "__main__":
    main(tyro.cli(Args))