#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

# DATASETS="bisect bisect_3d bisect_3d_prop bisect_6d"

# DATASETS="wine breastw annthyroid pima cardio glass ionosphere pendigits shuttle diabetes moodify "

DATASETS="bisect_3d_skewed"

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

for dataset in "${DATASET_ARRAY[@]}"; do

    DATASET_PATH="../data/syn/"

        python $SCRIPT_PATH \
            --dataset_name "$dataset" \
            --dataset_path $DATASET_PATH \
            --model "EIF+" \
            --interpretation "EXIFFI+" \
            --scenario 2 \
            --feature1 0 \
            --feature2 2

        # python $SCRIPT_PATH \
        #     --dataset_name "$dataset" \
        #     --dataset_path $DATASET_PATH \
        #     --model "EIF+" \
        #     --interpretation "EXIFFI+" \
        #     --scenario 1 \
        #     --feature1 0 \
        #     --feature2 2
done

# pre-process ONLY FOR REAL WORLD DATASET 
# --pre_process 1 \ 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --interpretation "EXIFFI+" \
    #     --scenario 1 \
    #     --feats_plot "(1,0)"

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --interpretation "EXIFFI" \
    #     --pre_process 1 \
    #     --scenario 2 \
    #     --feats_plot "(3,0)"

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --interpretation "EXIFFI" \
    #     --pre_process 1 \
    #     --scenario 1 \
    #     --feats_plot "(3,0)"

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "IF" \
    #     --interpretation "DIFFI" \
    #     --pre_process 1 \
    #     --scenario 2 \
    #     --feats_plot "(3,0)"

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "IF" \
    #     --interpretation "DIFFI" \
    #     --pre_process 1 \
    #     --scenario 1 \
    #     --feats_plot "(3,0)"

