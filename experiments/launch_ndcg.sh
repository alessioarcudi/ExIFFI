#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_ndcg.py"

# List of datasets

DATASETS="bisect bisect_3d bisect_3d_skewed bisect_6d"

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets
DATASET_PATH="../data/syn/"

for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "EXIFFI" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "EXIFFI" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "RandomForest" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "RandomForest" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "RandomForest" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "RandomForest" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "RandomForest" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "RandomForest" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "ECOD" \
        --interpretation "ECOD" \
        --scenario 1 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path $DATASET_PATH \
        --model "ECOD" \
        --interpretation "ECOD" \
        --scenario 2
    

done