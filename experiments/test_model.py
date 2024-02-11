"""
Python Script to launch the Model Comparison experiments.
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from append_dir import append_dirname
append_dirname("ExIFFI")

from glob import glob

from pyod.models.dif import DIF
from pyod.models.auto_encoder import AutoEncoder

from scipy.io import loadmat

sys.path.append('../src')
from src.performance_report_functions import *
from src.utils import *

sys.path.append('../models')
from models.forests import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Parse command line arguments
# In this function we can add all the parameters to modify to do the 
# ablation studies 
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--savedir", 
        type=str, 
        required=True, 
        help="Save directory for the results"
    )

    parser.add_argument(
        "--n_trees", 
        type=int, 
        default=300, 
        help="Number of trees in IF,EIF or EIF+"
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Contamination level for the dataset"
    )

    parser.add_argument(
        "--hidden_neurons",
        type=int,
        nargs='+',
        help="Number of neurons in the hidden layers of the AutoEncoder.All the values must be lower or equal to the number of features"
    )

    parser.add_argument(
        "--n_runs_imps",
        type=int,
        default=10,
        help="Set number of runs for the importance computation",
    )

    parser.add_argument(
    "--dataset_names",
    required=True,
    nargs="+",
    type=str,
    help="List of names of datasets on which to run the experiments",
    )

    parser.add_argument(
    "--model_names",
    required=True,
    nargs="+",
    type=str,
    help="List of names of models on which to run the experiments. Accepted values: ['EIF','EIF+','IF', 'DIF', 'INNE', 'AutoEncoder']",
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename of the output saved file. If None, it is automatically generated",
    )

    parser.add_argument(
        "--add_bash",
        action="store_true",
        help="If set, add bash -c to the command for timing the code",
    )
    
    return parser.parse_args()

# Use the model name str obtained from the command line and return the model object

def get_model(model_name):
    if model_name == "EIF":
        return ExtendedIsolationForest(n_estimators=args.n_trees,contamination=args.contamination,plus=0)
    elif model_name == "EIF+":
        return ExtendedIsolationForest(n_estimators=args.n_trees,contamination=args.contamination,plus=1)
    elif model_name == "IF":
        return IsolationForest(n_estimators=args.n_trees,contamination=args.contamination)
    elif model_name == "DIF":
        return DIF(n_estimators=args.n_trees,contamination=args.contamination)
    elif model_name == "AutoEncoder":
        return AutoEncoder(hidden_neurons=args.hidden_neurons,contamination=args.contamination)
    else:
        raise ValueError(f"Model {model_name} not supported")

# Evaluate the model performances 

def evaluate_model(model_name, X_train, X_test, y, name, save_dir, filename=None):

    # Fit the model
    print(f"Fitting {name} model")
    model=get_model(model_name)
    model.fit(X_train)

    # Compute the performance metrics
    print(f"Computing performance metrics for {name} model")
    if model_name=='IF':
        score=model.predict(X_test)
        perf=performance_if(y,score)
    elif model_name=='EIF' or model_name=='EIF+':
        score=model.predict(X_test)
        perf=performance_eif(y,score,X_test,model)

    return perf

def get_filename(dataset_name: str,partial_name="test_performance"):
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    partial_filename = (
        current_time + "_" + partial_name + "_" + dataset_name + ".npz"
    )
    return partial_filename

def main(args):
    path=os.getcwd()
    path = os.path.dirname(path)
    path_real = os.path.join(path, "data", "real")
    mat_files_real = glob(os.path.join(path_real, "*.mat"))
    mat_file_names_real = {os.path.basename(x).split(".")[0]: x for x in mat_files_real}
    csv_files_real = glob(os.path.join(path_real, "*.csv"))
    csv_file_names_real = {os.path.basename(x).split(".")[0]: x for x in csv_files_real}
    dataset_names = list(mat_file_names_real.keys()) + list(csv_file_names_real.keys())
    mat_file_names_real.update(csv_file_names_real)
    dataset_paths = mat_file_names_real.copy()

    dataset_names = args.dataset_names

    for name in dataset_names:
        print(f"Loading and pre processing {name} dataset")
        X_train,X_test,X=load_preprocess(name,dataset_paths[name])

        print('Shape of the datasets:')
        print(f'X_train: {X_train.shape}')
        print(f'X_test: {X_test.shape}')
        print(f'X: {X.shape}')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


"""
Things to add: 

At the end we want to obtain a script that is similar to test_parallel.py in the HPC-Project-AD
repository. 

- Add the `evaluate_performance` and `collect_performance` functions to compute multiple tests and 
    store the results in a dictionary that can successively go in the npz file 

- Add the execution time computation using time python command -> work with add_bash 

- Use the trick of the --wrapper option to run the experiment on different datasets as separate commands 

- Add a function to do the experiment on ExIFFI (an extension of the test_exiffi function)
    - `compute_global_importances`
    - `bar_plot`
    - `score_plot`
    - `importance_map`
    - `complete_importance_map`

- Add a function to do the experiments with the Feature Selection Proxy Task 

"""







