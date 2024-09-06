#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

# List of datasets

DATASETS="bisect_3d_prop"

# DATASETS="wine breastw annthyroid pima cardio glass ionosphere pendigits shuttle diabetes moodify "

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets
DATASET_PATH="../data/syn/"

for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --scenario 2 
        
    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "EXIFFI" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "EXIFFI" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "RandomForest" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "RandomForest" \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "RandomForest" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "RandomForest" \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "RandomForest" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "RandomForest" \
        --scenario 1
        
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
