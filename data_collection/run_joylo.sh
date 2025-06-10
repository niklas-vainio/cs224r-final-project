OG_GELLO_DIR="../external/og-gello"

# Replace with a config file generated with calibrate_joints.py:
# JOINT_CONFIG_FILE="joint_config_jellyfish_pro.yaml"
JOINT_CONFIG_FILE="joint_config_jellyfish_pro.yaml"

python "${OG_GELLO_DIR}/experiments/run_joylo.py" --joint_config_file ${JOINT_CONFIG_FILE}