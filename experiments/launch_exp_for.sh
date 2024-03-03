#!/bin/bash

# Path to the Python script to execute
# SCRIPT_PATH="test_global_importancies.py"
#SCRIPT_PATH="test_feature_selection.py"
#SCRIPT_PATH="test_contamination_precision.py"
SCRIPT_PATH="test_GFI_FS.py"

# List of datasets
#DATASETS="Xaxis Yaxis bisect bisect_3d bisect_6d"
DATASETS="wine glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
#DATASETS="Yaxis bisect bisect_3d bisect_6d"
#DATASETS="ionosphere"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasetsok, put the `xlim` at 1.1
DATASET_PATH="../data/real/"

# Iterate over the datasets and call the Python command for each dataset
for dataset in "${DATASET_ARRAY[@]}"; do
    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --pre_process \
        --model "EIF" \
        --interpretation "RandomForest" \
        --scenario 1 \
        --include_random \
        --downsample \

done
