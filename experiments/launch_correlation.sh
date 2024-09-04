#!/bin/bash

SCRIPT_PATH="test_correlation.py"

DATASETS="bisect bisect_3d_prop"

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Next datasets → shuttle moodify

DATASET_PATH="../data/syn/"

for dataset in "${DATASET_ARRAY[@]}"; do

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
                --scenario 2 

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
                --interpretation "DIFFI" \
                --scenario 2 
done

# python $SCRIPT_PATH \
#         --dataset_name $dataset \
#         --dataset_path $DATASET_PATH \
#         --model "EIF+" \
#         --interpretation "EIF+_RandomForest" \
#         --scenario 2 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF" \
#         --interpretation "EIF_RandomForest" \
#         --scenario 2 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "IF" \
#         --interpretation "IF_RandomForest" \
#         --scenario 2