#!/bin/bash

SCRIPT_PATH="test_eifplus_eta.py"

DATASETS="breastw"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets 
DATASET_PATH="../data/real/"

for dataset in "${DATASET_ARRAY[@]}"; do

     python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF+" \
        --change_ylim 1


done