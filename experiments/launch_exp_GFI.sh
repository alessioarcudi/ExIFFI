#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

# List of datasets

# DATASETS="breastw"

DATASETS="wine breastw annthyroid pima cardio glass ionosphere pendigits shuttle diabetes moodify "

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets
DATASET_PATH="../data/real/"

for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "ECOD" \
        --interpretation "ECOD" \
        --scenario 2 \
        --n_runs 1 \
        --percentile 0.99 \
        --pre_process

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "ECOD" \
        --interpretation "ECOD" \
        --scenario 1 \
        --n_runs 1 \
        --percentile 0.99 \
        --pre_process

done

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF+" \
#     --interpretation "EXIFFI+" \
#     --scenario 2

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF+" \
#     --interpretation "EXIFFI+" \
#     --scenario 1

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF" \
#     --interpretation "EXIFFI" \
#     --scenario 2

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF" \
#     --interpretation "EXIFFI" \
#     --scenario 1

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "IF" \
#     --interpretation "DIFFI" \
#     --scenario 1 

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "IF" \
#     --interpretation "DIFFI" \
#     --scenario 2

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "IF" \
#     --interpretation "EXIFFI" \
#     --scenario 1 

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "IF" \
#     --interpretation "EXIFFI" \
#     --scenario 2

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF+" \
#     --interpretation "RandomForest" \
#     --scenario 2

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF+" \
#     --interpretation "RandomForest" \
#     --scenario 1

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF" \
#     --interpretation "RandomForest" \
#     --scenario 2

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF" \
#     --interpretation "RandomForest" \
#     --scenario 1

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "IF" \
#     --interpretation "RandomForest" \
#     --scenario 2

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "IF" \
#     --interpretation "RandomForest" \
#     --scenario 1



# Use pre_process ONLY ON THE REAL WORLD DATASETS

# --pre_process 
