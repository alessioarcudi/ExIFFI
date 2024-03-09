#!/bin/bash

# Path to the Python script to execute
#SCRIPT_PATH="test_global_importancies.py"
SCRIPT_PATH="test_feature_selection.py"
#SCRIPT_PATH="test_GFI_FS.py"
#SCRIPT_PATH="test_local_importances.py"

# List of datasets
#DATASETS="Xaxis Yaxis bisect bisect_3d bisect_6d"
#DATASETS="glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
#DATASETS="Yaxis bisect bisect_3d bisect_6d"
DATASETS="bisect"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets 
DATASET_PATH="../data/syn/"

# Iterate over the datasets and call the Python command for each dataset
for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "RandomForest" \
        --pre_process \
        --scenario 1  

done

#--feats_plot "(10,0)" \
#
