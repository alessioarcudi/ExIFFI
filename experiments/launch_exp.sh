#!/bin/bash


# Path to the Python script to execute 
#SCRIPT_PATH="test_global_importancies.py"
#SCRIPT_PATH="test_feature_selection.py"
SCRIPT_PATH="test_GFI_FS.py"

#DATASETS="Xaxis"
DATASETS="wine glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify"


DATASET_PATHS="../data/syn/"

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --pre_process \
    --model "EIF" \
    --interpretation "RandomForest" \
    --box_loc "(3,0.8)"
