#!/bin/bash

printf "Executing DIFFI Experiments:\n"

# Path to the Python script to execute 
SCRIPT_PATH="test_DIFFI_importances.py"

#DATASETS="wine glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
DATASETS="glass"

DATASET_PATHS="../data/real/"

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --n_runs 10 \
    --split True \
    --n_estimators 300 \
    --feats_plot '(5, 4)'


# Possible parameters to add (for now they are at the default value)
# --dataset_path (if different from ../data/real/)
# --max_samples 10 \
# --train_size 0.7 \
# --contamination 0.05 \ 
