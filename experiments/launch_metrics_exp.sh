#!/bin/bash

SCRIPT_PATH="test_metrics.py"


# List of datasets
DATASETS="bisect_3d_skewed"
#DATASETS="glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify"
#DATASETS="Yaxis bisect bisect_3d bisect_6d"
# DATASETS="breastw ionosphere annthyroid pendigits diabetes shuttle moodify"

# Split the DATASETS string into an array
IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets 
DATASET_PATH="../data/syn/"

# Iterate over the datasets and call the Python command for each dataset
for dataset in "${DATASET_ARRAY[@]}"; do
 
    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF+" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF+" \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "EIF" \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "IF" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "IF" \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "DIF" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "DIF" \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "AnomalyAutoencoder" \
        --scenario 2

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "AnomalyAutoencoder" \
        --scenario 1

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "ECOD" \
        --scenario 2 

    python $SCRIPT_PATH \
        --dataset_name "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --model "ECOD" \
        --scenario 1 

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