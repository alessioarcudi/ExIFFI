#!/bin/bash

SCRIPT_PATH="get_metrics.py"

DATASETS="shuttle"

DATASET_PATH="../data/real/"

MODELS="EIF+ EIF IF DIF AnomalyAutoencoder"
IFS=' ' read -ra MODELS_ARRAY <<< "$MODELS"
SCENARIOS=(2 1)

for model in "${MODELS_ARRAY[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do

       python $SCRIPT_PATH \
            --dataset_name $DATASETS \
            --dataset_path $DATASET_PATH \
            --model "$model" \
            --scenario "$scenario"

    done
done