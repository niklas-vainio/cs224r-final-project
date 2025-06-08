# Run the eval script

POLICY_CHECKPOINT="/home/zhenyu/b1k_project/cs224r-final-project/checkpoints/last.pth"

OMNIGIBSON_HEADLESS=1 python rollout_and_eval_policy.py \
--num_episodes=50 \
--policy_checkpoint=${POLICY_CHECKPOINT}\

# python rollout_and_eval_policy.py \
# --num_episodes=1 \
# --policy_checkpoint=${POLICY_CHECKPOINT}\