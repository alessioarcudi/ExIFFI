#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

# List of datasets
DATASETS="bisect_3d_prop"

# Path to the datasets
DATASET_PATH="../data/syn/"


python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "EIF+" \
    --interpretation "EXIFFI+" \
    --scenario 2 

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "IF" \
#     --interpretation "DIFFI" \
#     --scenario 1 \
#     --pre_process 

# Use pre_process ONLY ON THE REAL WORLD DATASETS

# --pre_process 
