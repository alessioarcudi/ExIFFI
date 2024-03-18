#!/bin/bash

SCRIPT_PATH="test_time_scaling.py"

DATASETS="Xaxis_250000_6"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets 
DATASET_PATH="../data/syn/"

# Iterate over the datasets and call the Python command for each dataset
for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --compute_GFI 1 
        
done

# --interpretation "DIFFI" \
# --compute_GFI 1 