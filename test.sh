#!/bin/bash
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"
TRAINIDS_DIR="${TEST_ROOT}/labelTrainIds"

source ~/venv/CISS_test/bin/activate
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
# Inference.
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --eval-options imgfile_prefix=${TRAINIDS_DIR} to_label_id=False --show-dir ${SHOW_DIR} --opacity 1
# Evaluation.
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU
