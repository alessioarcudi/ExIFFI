#!/bin/bash


# Path to the Python script to execute 
SCRIPT_PATH="test_contamination_precision.py"

# Dataset 
DATASETS="Xaxis"

# Path to the datasets
DATASET_PATHS="../data/syn/"

# Experiment EIF+

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATHS \
#     --model "EIF+" \
#     --interpretation "EXIFFI+" \
#     --compute_GFI 1 

# Experiment EIF

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "EIF" \
    --interpretation "EXIFFI" \
    --compute_GFI 1 \

# Experiment IF

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "IF" \
    --interpretation "DIFFI" \
    --compute_GFI 1 \

# Experiment DIF

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "DIF" \

# Experiment AnomalyAutoencoder

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATHS \
    --model "AnomalyAutoencoder" \
