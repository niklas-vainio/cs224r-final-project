# Run the eval script

POLICY_CHECKPOINT="/home/zhenyu/b1k_project/cs224r-final-project/checkpoints/run2_ckpt/last.pth"

python rollout_and_eval_policy.py \
--num_episodes=1 \
--policy_checkpoint=${POLICY_CHECKPOINT}\