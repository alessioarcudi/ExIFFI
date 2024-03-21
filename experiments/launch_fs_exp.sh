#!/bin/bash

#SCRIPT_PATH="test_metrics.py"
SCRIPT_PATH="test_feature_selection.py"
#SCRIPT_PATH="test_global_importancies.py"

DATASETS="moodify"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets 
DATASET_PATH="../data/real/"

# Iterate over the datasets and call the Python command for each dataset
for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --model_interpretation "EIF+" \
        --interpretation "EXIFFI+" \
        --pre_process \
        --scenario 1 \
        --compute_random

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --model_interpretation "EIF" \
        --interpretation "EXIFFI" \
        --pre_process \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "DIFFI" \
        --pre_process \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --model_interpretation "EIF+" \
        --interpretation "RandomForest" \
        --pre_process \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --model_interpretation "EIF" \
        --interpretation "RandomForest" \
        --pre_process \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "RandomForest" \
        --pre_process \
        --scenario 1
done
