#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

# List of datasets
DATASETS="moodify"


# Path to the datasets
DATASET_PATH="../data/real/"


python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "IF" \
    --interpretation "DIFFI" \
    --scenario 2 \
    --pre_process

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "IF" \
    --interpretation "DIFFI" \
    --scenario 1 \
    --pre_process 

# Use pre_process ONLY ON THE NON SYNTHETIC DATASETS

# --pre_process 
