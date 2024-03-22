#!/bin/bash


# Path to the Python script to execute 
SCRIPT_PATH="test_contamination_precision.py"

# Dataset 
DATASETS="diabetes"

# Path to the datasets
DATASET_PATHS="../data/real/"

# Experiment EIF+

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "EIF+" \
    --interpretation "EXIFFI+" \
    --pre_process 1 \
    --compute_GFI 1 

# Experiment EIF

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "EIF" \
    --interpretation "EXIFFI" \
    --pre_process 1 \
    --compute_GFI 1 

# # Experiment IF

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "IF" \
    --interpretation "DIFFI" \
    --pre_process 1 \
    --compute_GFI 1 

# # Experiment DIF

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --pre_process 1 \
    --model "DIF" 

# # Experiment AnomalyAutoencoder

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --pre_process 1 \
    --model "AnomalyAutoencoder" 
