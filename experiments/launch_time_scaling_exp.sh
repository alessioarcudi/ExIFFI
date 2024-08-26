#!/bin/bash

SCRIPT_PATH="test_time_scaling.py"

DATASETS="Xaxis_5000_6"

DATASET_PATH="../data/syn/"


python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "KernelSHAP" \
        --compute_GFI 1 \
        --background 1.0

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF" \
#         --interpretation "EXIFFI" \
#         --compute_fit_predict 1 \
#         --compute_GFI 1 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "IF" \
#         --interpretation "DIFFI" \
#         --compute_fit_predict 1 \
#         --compute_GFI 1 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF+_RF" \
#         --interpretation "RandomForest" \
#         --compute_GFI 1 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "DIF" \
#         --compute_fit_predict 1 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "AnomalyAutoencoder" \
#         --compute_fit_predict 1 