#!/bin/bash

# Path to the Python script to execute
#SCRIPT_PATH="test_global_importancies.py"
#SCRIPT_PATH="test_feature_selection.py"
#SCRIPT_PATH="test_metrics.py"
#SCRIPT_PATH="test_GFI_FS.py"
SCRIPT_PATH="test_local_importances.py"
#SCRIPT_PATH="test_time_scaling.py"

# List of datasets
#DATASETS="Xaxis Yaxis bisect bisect_3d bisect_6d"
#DATASETS="glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
#DATASETS="Yaxis bisect bisect_3d bisect_6d"
DATASETS="annthyroid"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets 
DATASET_PATH="../data/real/"

# Iterate over the datasets and call the Python command for each dataset
for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --pre_process "True" \
        --feats_plot "(1,3)" \
        --scenario 2
done

# --interpretation "DIFFI" \
# --compute_GFI 1

# For local scoremaps experiments 
#--feats_plot "(10,0)" \
#

# To use the pre-processing step
#--pre_process \

# For Feature Selection experiments
# --model_interpretation "IF" \
# --interpretation "RandomForest" \

# For Performance Metrics experiments

# --interpretation "EXIFFI+" \
#         --feats_plot "(1,5)" \
#         --pre_process \