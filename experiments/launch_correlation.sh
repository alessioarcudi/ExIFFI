#!/bin/bash

SCRIPT_PATH="test_correlation.py"

DATASETS="annthyroid breastw cardio diabetes glass ionosphere moodify pendigits pima shuttle wine"

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Next datasets → shuttle moodify

DATASET_PATH="../data/real/"

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
                --model "EIF" \
                --interpretation "EXIFFI" \
                --scenario 1 

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
                --interpretation "DIFFI" \
                --scenario 1
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