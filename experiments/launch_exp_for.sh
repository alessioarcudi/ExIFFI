#!/bin/bash

# Path to the Python script to execute
# SCRIPT_PATH="test_global_importancies.py"
# SCRIPT_PATH="test_feature_selection.py"
SCRIPT_PATH="test_GFI_FS.py"

# List of datasets
#DATASETS="wine"
DATASETS="bisect bisect_3d bisect_6d"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets
DATASET_PATH="../data/syn/"

# Iterate over the datasets and call the Python command for each dataset
for dataset in "${DATASET_ARRAY[@]}"; do
    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --pre_process \
        --model "EIF" \
        --interpretation "RandomForest" \
        --scenario 2 \

done
