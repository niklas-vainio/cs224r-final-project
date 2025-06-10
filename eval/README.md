# Evaluation

This directory contains scripts for evaluating a trained policy in the OmniGibson environment. The provided `policy.pth` checkpoint is from the end of training process.

To run evaluation, replace the `POLICY_CHECKPOINT` variable in `run_eval.sh` and then run:
```
./run_eval.sh
```
This will create a `.json` file for each episode.

To run evaluation for a different scene/task, you will need to export your own `env_config.json` and `scene_config.json` files.

**Refer to the installation instructions in `og-gello` to configure an environment with OmniGibson and its dependencies, and in `brs-algo` to install the dependencies needed to run the WB-VIMA policy.**