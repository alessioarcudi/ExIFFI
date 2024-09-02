#!/bin/bash

SCRIPT_PATH="test_time_scaling.py"

DATASET="bisect_3d_prop"

MODELS="ECOD"

# Split the DATASETS string into an array
IFS=' ' read -ra MODEL_ARRAY <<< "$MODELS"

# Path to the datasets 
DATASET_PATH="../data/syn/"

# Iterate over the datasets and call the Python command for each dataset
for model in "${MODEL_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name $DATASET \
        --dataset_path "$DATASET_PATH" \
        --model "$model" \
        --compute_fit_predict 1
        
done

# --interpretation "DIFFI" \
# --compute_GFI 1 