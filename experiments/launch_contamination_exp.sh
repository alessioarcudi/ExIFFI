#!/bin/bash


# Path to the Python script to execute 
SCRIPT_PATH="test_contamination_precision.py"

# Dataset 
# DATASETS="breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
DATASETS="moodify"

# Path to the datasets
DATASET_PATHS="../data/real/"

# Experiment ECOD

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "ECOD" 
    # --change_ylim 1.1 

# # Experiment EIF+

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATHS \
#     --model "EIF+" \
#     --interpretation "EXIFFI+" \
#     --compute_GFI 1 

# # Experiment EIF

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATHS \
#     --model "EIF" \
#     --interpretation "EXIFFI" \
#     --compute_GFI 1 

# # Experiment IF

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATHS \
#     --model "IF" \
#     --interpretation "DIFFI" \
#     --compute_GFI 1 

# # Experiment DIF

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATHS \
#     --model "DIF" 

# # Experiment AnomalyAutoencoder

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATHS \
#     --model "AnomalyAutoencoder" 
