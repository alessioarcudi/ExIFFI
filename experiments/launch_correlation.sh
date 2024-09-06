#!/bin/bash

SCRIPT_PATH="test_correlation.py"

# DATASETS="Xaxis bisect bisect_3d bisect_3d_prop bisect_6d"

# PUT pre_process FOR THE REAL DATASETS 
DATASETS="annthyroid breastw cardio diabetes glass ionosphere moodify pendigits pima shuttle wine"

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Next datasets → shuttle moodify

DATASET_PATH="../data/real/"

for dataset in "${DATASET_ARRAY[@]}"; do

        python $SCRIPT_PATH \
                --dataset_name "$dataset" \
                --dataset_path $DATASET_PATH \
                --model "ECOD" \
                --interpretation "ECOD" \
                --scenario 2 \
                --n_runs 1 \
                --pre_process 1

        python $SCRIPT_PATH \
                --dataset_name "$dataset" \
                --dataset_path $DATASET_PATH \
                --model "ECOD" \
                --interpretation "ECOD" \
                --scenario 1 \
                --n_runs 1 \
                --pre_process 1
done

# python $SCRIPT_PATH \
#                 --dataset_name "$dataset" \
#                 --dataset_path $DATASET_PATH \
#                 --model "EIF" \
#                 --interpretation "EXIFFI" \
#                 --scenario 1 

# python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path $DATASET_PATH \
#         --model "IF" \
#         --interpretation "EXIFFI" \
#         --scenario 1 

# python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path $DATASET_PATH \
#         --model "IF" \
#         --interpretation "DIFFI" \
#         --scenario 1

# python $SCRIPT_PATH \
#         --dataset_name $dataset \
#         --dataset_path $DATASET_PATH \
#         --model "EIF+" \
#         --interpretation "EIF+_RandomForest" \
#         --scenario 2 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF" \
#         --interpretation "EIF_RandomForest" \
#         --scenario 2 

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "IF" \
#         --interpretation "IF_RandomForest" \
#         --scenario 2