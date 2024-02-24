#!/bin/bash

printf "Executing Experiments:\n"

# Path to the Python script to execute 
#SCRIPT_PATH="test_global_importancies.py"
SCRIPT_PATH="test_feature_selection.py"

#DATASETS="wine glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
DATASETS="shuttle"

DATASET_PATHS="../data/real/"

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --pre_process \
    --model "EIF+" 


# Possible parameters to add (for now they are at the default value)
# --dataset_path (if different from ../data/real/)
# --max_samples 10 \
# --train_size 0.7 \
# --contamination 0.05 \ 
