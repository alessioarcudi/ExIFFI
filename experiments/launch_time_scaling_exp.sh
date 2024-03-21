#!/bin/bash

SCRIPT_PATH="test_time_scaling.py"

DATASETS="wine"

DATASET_PATH="../data/real/"

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+_RF" \
        --interpretation "RandomForest" \
        --compute_GFI 1 