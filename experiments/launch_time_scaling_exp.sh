#!/bin/bash

SCRIPT_PATH="test_time_scaling.py"

# Varying number of samples 
# DATASETS="Xaxis_100_6"
# DATASETS="Xaxis_250_6"
# DATASETS="Xaxis_500_6"
# DATASETS="Xaxis_1000_6"
# DATASETS="Xaxis_2500_6" 
# DATASETS="Xaxis_5000_6"
# DATASETS="Xaxis_10000_6"
# DATASETS="Xaxis_25000_6"
# DATASETS="Xaxis_50000_6"
# DATASETS="Xaxis_100000_6"

# Varying number of features

DATASETS="Xaxis_5000_16"
# DATASETS="Xaxis_5000_32"
# DATASETS="Xaxis_5000_64"
# DATASETS="Xaxis_5000_128"
# DATASETS="Xaxis_5000_256"
# DATASETS="Xaxis_5000_512"

DATASET_PATH="../data/syn/"


python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "KernelSHAP" \
        --compute_GFI 1 \
        --background 0.1

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