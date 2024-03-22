#!/bin/bash

SCRIPT_PATH="test_global_importancies.py"

DATASETS="breastw"

DATASET_PATHS="../data/real/"

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "IF" \
    --interpretation "DIFFI" \
    --pre_process \
    --scenario 2