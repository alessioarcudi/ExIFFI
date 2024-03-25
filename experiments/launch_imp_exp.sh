#!/bin/bash

SCRIPT_PATH="test_global_importancies.py"

DATASETS="bisect"

DATASET_PATH="../data/syn/"

SCENARIOS=(1 2)

for scenario in "${SCENARIOS[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario "$scenario" 

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --scenario "$scenario" 

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --scenario "$scenario" 

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "RandomForest" \
        --scenario "$scenario"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "RandomForest" \
        --scenario "$scenario" 

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "RandomForest" \
        --scenario "$scenario"

done
