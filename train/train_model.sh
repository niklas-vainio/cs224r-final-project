#!/bin/bash

TASK=picking_up_trash
DATA_ROOT_DIR="data/merged_data.hdf5"
CFG_DIR="cfg"
BASE_TRAIN_CFG="cfg.yaml"
BRS_DIR="external/brs-algo"

WANDB_PROJECT="cs224r-final-project"

# Run training
python ${BRS_DIR}/main/train/train.py \
--config-path=${CFG_DIR} \
--config-name=${BASE_TRAIN_CFG} \
bs=128 \
task=${TASK} \
gpus=1 \
use_wandb=true \
wandb_project=${WANDB_PROJECT} \
exp_root_dir="experiments" \
data_dir=${DATA_ROOT_DIR}