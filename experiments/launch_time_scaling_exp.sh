#!/bin/bash

SCRIPT_PATH="test_time_scaling.py"

DATASETS="diabetes"

DATASET_PATH="../data/real/"

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+_RF" \
        --interpretation "RandomForest" \
        --pre_process \
        --compute_GFI 1 