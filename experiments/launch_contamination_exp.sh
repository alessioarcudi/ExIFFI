#!/bin/bash

# Path to the Python script to execute 
SCRIPT_PATH="test_contamination_precision.py"

# Dataset 
# DATASETS="breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
DATASETS="bisect_3d_skewed"

# Path to the datasets
DATASET_PATHS="../data/syn/"

# Experiment EIF+

python3 $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "EIF+" \
    --interpretation "EXIFFI+" \
    --compute_GFI 1 

# Experiment EIF

python3 $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "EIF" \
    --interpretation "EXIFFI" \
    --compute_GFI 1 

# Experiment IF

python3 $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "IF" \
    --interpretation "DIFFI" \
    --compute_GFI 1 

# Experiment DIF

python3 $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "DIF" 

# Experiment AnomalyAutoencoder

python3 $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "AnomalyAutoencoder" 

# Experiment ECOD

python3 $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "ECOD" \
    --interpretation "ECOD" \
    --compute_GFI 1
