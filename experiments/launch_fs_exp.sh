#!/bin/bash

SCRIPT_PATH="test_feature_selection.py"

# DATASETS="bisect"

DATASETS="bisect_3d_prop"

# DATASETS="wine breastw annthyroid pima cardio glass ionosphere pendigits shuttle diabetes moodify "

IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Path to the datasets 
DATASET_PATH="../data/syn/"

for dataset in "${DATASET_ARRAY[@]}"; do

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario 1 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario 2 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF" \
        --interpretation "EXIFFI" \
        --scenario 1 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF" \
        --interpretation "EXIFFI" \
        --scenario 2 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "EXIFFI" \
        --scenario 1 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "EXIFFI" \
        --scenario 2 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "DIFFI" \
        --scenario 1 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "DIFFI" \
        --scenario 2 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "RandomForest" \
        --scenario 1 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "IF" \
        --interpretation "RandomForest" \
        --scenario 2 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF" \
        --interpretation "RandomForest" \
        --scenario 1 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF" \
        --interpretation "RandomForest" \
        --scenario 2 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF+" \
        --interpretation "RandomForest" \
        --scenario 1 \
        --change_ylim

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --model_interpretation "EIF+" \
        --interpretation "RandomForest" \
        --scenario 2 \
        --change_ylim

done


 # python $SCRIPT_PATH \
    #     --dataset_name "$dataset" \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "ECOD" \
    #     --interpretation "ECOD" \
    #     --scenario 2 \
    #     --n_runs 1 \
    #     --change_ylim  
 
    # python $SCRIPT_PATH \
    #     --dataset_name "$dataset" \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "ECOD" \
    #     --interpretation "ECOD" \
    #     --scenario 1 \
    #     --n_runs 1 \
    #     --change_ylim  

    # python $SCRIPT_PATH \
    #     --dataset_name "$dataset" \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --model_interpretation "ECOD" \
    #     --interpretation "ECOD" \
    #     --scenario 2 \
    #     --n_runs 1 \
    #     --change_ylim  
 
    # python $SCRIPT_PATH \
    #     --dataset_name "$dataset" \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --model_interpretation "ECOD" \
    #     --interpretation "ECOD" \
    #     --scenario 1 \
    #     --n_runs 1 \
    #     --change_ylim  
 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --model_interpretation "IF" \
    #     --interpretation "DIFFI" \
    #     --scenario 1 \


    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "EIF+" \
    #     --interpretation "EXIFFI+" \
    #     --scenario 2 \
    #     --compute_random

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "EIF" \
    #     --interpretation "EXIFFI" \
    #     --scenario 2 \
    #     --change_ylim

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "IF" \
    #     --interpretation "DIFFI" \
    #     --scenario 2 \

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "IF" \
    #     --interpretation "EXIFFI" \
    #     --scenario 2 \
    #     --change_ylim

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "EIF+" \
    #     --interpretation "RandomForest" \
    #     --scenario 2 \
    #     --change_ylim

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "EIF" \
    #     --interpretation "RandomForest" \
    #     --scenario 2 \
    #     --change_ylim

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "IF" \
    #     --interpretation "RandomForest" \
    #     --scenario 2 \
    #     --change_ylim

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --model_interpretation "IF" \
    #     --interpretation "RandomForest" \
    #     --scenario 1 \
    #     --change_ylim 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --model_interpretation "EIF" \
    #     --interpretation "RandomForest" \
    #     --scenario 1 \
    #     --change_ylim  

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --model_interpretation "EIF+" \
    #     --interpretation "RandomForest" \
    #     --scenario 1 \
    #     --change_ylim 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --model_interpretation "IF" \
    #     --interpretation "EXIFFI" \
    #     --scenario 1 \
    #     --change_ylim

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --model_interpretation "EIF+" \
    #     --interpretation "EXIFFI+" \
    #     --scenario 2 



# Split the DATASETS string into an array
# IFS=' ' read -ra DATASET_ARRAY <<< "$DATASETS"

# Iterate over the datasets and call the Python command for each dataset
# for dataset in "${DATASET_ARRAY[@]}"; do

#     python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path "$DATASET_PATH" \
#         --model "EIF" \
#         --model_interpretation "EIF+" \
#         --interpretation "EXIFFI+" \
#         --pre_process \
#         --scenario 1 \
#         --compute_random 

#     python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path "$DATASET_PATH" \
#         --model "EIF" \
#         --model_interpretation "EIF" \
#         --interpretation "EXIFFI" \
#         --pre_process \
#         --scenario 1 

#     python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path "$DATASET_PATH" \
#         --model "EIF" \
#         --model_interpretation "IF" \
#         --interpretation "DIFFI" \
#         --pre_process \
#         --scenario 1 

#     python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path "$DATASET_PATH" \
#         --model "EIF" \
#         --model_interpretation "EIF+" \
#         --interpretation "RandomForest" \
#         --pre_process \
#         --scenario 1

#     python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path "$DATASET_PATH" \
#         --model "EIF" \
#         --model_interpretation "EIF" \
#         --interpretation "RandomForest" \
#         --pre_process \
#         --scenario 1 

#     python $SCRIPT_PATH \
#         --dataset_name "$dataset" \
#         --dataset_path "$DATASET_PATH" \
#         --model "EIF" \
#         --model_interpretation "IF" \
#         --interpretation "RandomForest" \
#         --pre_process \
#         --scenario 1

# done

# MODELS="EIF+ EIF"
# IFS=' ' read -ra MODELS_ARRAY <<< "$MODELS"
# SCENARIOS=(1 2)

# for model in "${MODELS_ARRAY[@]}"; do
#     for scenario in "${SCENARIOS[@]}"; do

#         python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "$model" \
#             --model_interpretation "EIF+" \
#             --interpretation "EXIFFI+" \
#             --scenario "$scenario" \
#             --compute_random 

#         python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "$model" \
#             --model_interpretation "EIF" \
#             --interpretation "EXIFFI" \
#             --scenario "$scenario" 

#         python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "$model" \
#             --model_interpretation "IF" \
#             --interpretation "DIFFI" \
#             --scenario "$scenario" 

#         python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "$model" \
#             --model_interpretation "EIF+" \
#             --interpretation "RandomForest" \
#             --scenario "$scenario"

#         python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "$model" \
#             --model_interpretation "EIF" \
#             --interpretation "RandomForest" \
#             --scenario "$scenario" 

#         python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "$model" \
#             --model_interpretation "IF" \
#             --interpretation "RandomForest" \
#             --scenario "$scenario"

#     done
# done

# python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "EIF+" \
#             --model_interpretation "EIF+" \
#             --interpretation "EXIFFI+" \
#             --pre_process \
#             --scenario 2

# python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "EIF" \
#             --model_interpretation "EIF" \
#             --interpretation "RandomForest" \
#             --pre_process \
#             --scenario 2

