# Run the eval script

# Replace with your checkpoint:
POLICY_CHECKPOINT="policy.pth"


# Use this version to run with no display (e.g. over SSH):
# 
# OMNIGIBSON_HEADLESS=1 python rollout_and_eval_policy.py \
# --num_episodes=1000 \
# --policy_checkpoint=${POLICY_CHECKPOINT}\

python rollout_and_eval_policy.py \
--num_episodes=100 \
--policy_checkpoint=${POLICY_CHECKPOINT}\