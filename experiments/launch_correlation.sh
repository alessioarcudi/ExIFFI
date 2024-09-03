#!/bin/bash

SCRIPT_PATH="test_correlation.py"

DATASETS="moodify"

# Next datasets → shuttle moodify

DATASET_PATH="../data/real/"

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario 2 

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --scenario 2 

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "EXIFFI" \
        --scenario 2 

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --scenario 2 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
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