#!/bin/bash

SCRIPT_PATH="test_time_scaling.py"

DATASETS="shuttle"

DATASET_PATH="../data/real/"

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF+_RF" \
#         --interpretation "RandomForest" \
#         --pre_process \
#         --compute_GFI 1 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF+" \
#         --interpretation "EXIFFI+" \
#         --pre_process \
#         --compute_fit_predict 1 \
#         --compute_GFI 1 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF" \
#         --interpretation "EXIFFI" \
#         --pre_process \
#         --compute_fit_predict 1 \
#         --compute_GFI 1 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "IF" \
#         --interpretation "DIFFI" \
#         --pre_process \
#         --compute_fit_predict 1 \
#         --compute_GFI 1 

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+_RF" \
        --interpretation "RandomForest" \
        --pre_process \
        --compute_GFI 1 

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "DIF" \
        --pre_process \
        --compute_fit_predict 1 

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "AnomalyAutoencoder" \
        --pre_process \
        --compute_fit_predict 1 