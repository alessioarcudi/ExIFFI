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

    # Set the contamination factor in the training set
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Contamination level for the dataset"
    )

    # Set the number of trees in the forest
    parser.add_argument(
    "--n_trees", 
    type=int, 
    default=300, 
    help="Number of trees in IF,EIF or EIF+, default 300"
    )

    # Directory to save the results 
    parser.add_argument(
        "--savedir", 
        type=str, 
        required=True, 
        help="Save directory for the results"
    )

    # Set the number of runs for the metrics computation in collect_precisions
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of runs for the GFI computation, default 10"
    )

    # List of datasets to be used in the experiments
    parser.add_argument(
    "--dataset_names",
    required=True,
    nargs="+",
    type=str,
    help="List of names of datasets on which to run the experiments",
    )

    # Set the seed 
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set seed for reproducibility, default None"
    )

    # Set the hidden neurons for the AutoEncoder
    parser.add_argument(
        "--hidden_neurons",
        type=int,
        nargs='+',
        help="Number of neurons in the hidden layers of the AutoEncoder.All the values must be lower or equal to the number of features"
    )

    # Scaler to use for the data normalization
    parser.add_argument(
        "--scaler",
        type=str,
        default="StandardScaler",
        help="""Scaler to use for the data normalization. 
        Accepted values: ['StandardScaler','MinMaxScaler','MaxAbsScaler','RobustScaler'],
        default StandardScaler"""
    )

    # Distribution to use for the selection of point p in the cutting hyperplanes
    parser.add_argument(
        "--distribution",
        type=str,
        default="normal_mean",
        help="""Distribution to use for the selection of point p in the cutting hyperplanes.
        Accepted values: ['normal_mean','normal_median','scaled_uniform'],
        default normal_mean"""
    )

    # Scaling factor eta in the distribution for intercept p 
    parser.add_argument(
        "--eta",
        type=int,
        default=2,
        help="""Scaling factor used in the definition of the distribution of the intercept p in the cutting hyperplanes,
        default 2"""
    )

    # Set the model names to be used in the experiments
    parser.add_argument(
    "--model_names",
    nargs="+",
    type=str,
    default='EIF+',
    help="""List of names of models on which to run the experiments.
    Accepted values: ['EIF','EIF+','IF', 'DIF', 'AutoEncoder'],
    default EIF+""",
    )

    # Run the wrapper for timing the code
    parser.add_argument(
    "--wrapper",
    action="store_true",
    help="If set, run the wrapper for timing the code",
    )

    # add bash -c to the command for timing the code
    parser.add_argument(
    "--add_bash",
    action="store_true",
    help="If set, add bash -c to the command for timing the code",
    )
    
    # Set the filename of the output saved file
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="""Filename of the output saved file. If None, it is automatically generated,
        default None""",
    )

    return parser.parse_args()

# Use the model name str obtained from the command line and return the model object

def get_model(model_name="EIF+"):
    if model_name == "EIF+":
        return ExtendedIsolationForest(n_estimators=args.n_trees,contamination=args.contamination,plus=1)
    elif model_name == "EIF":
        return ExtendedIsolationForest(n_estimators=args.n_trees,contamination=args.contamination,plus=0)
    elif model_name == "IF":
        return IsolationForest(n_estimators=args.n_trees,contamination=args.contamination)
    elif model_name == "DIF":
        return DIF(n_estimators=args.n_trees,contamination=args.contamination)
    elif model_name == "AutoEncoder":
        return AutoEncoder(hidden_neurons=args.hidden_neurons,contamination=args.contamination)
    else:
        raise ValueError(f"Model {model_name} not supported")

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
    path_syn = os.path.join(path, "data", "syn")
    mat_files_real = glob(os.path.join(path_real, "*.mat"))
    mat_file_names_real = {os.path.basename(x).split(".")[0]: x for x in mat_files_real}
    mat_files_syn = glob(os.path.join(path_syn, "*.mat"))
    mat_file_names_syn = {os.path.basename(x).split(".")[0]: x for x in mat_files_syn}
    csv_files_real = glob(os.path.join(path_real, "*.csv"))
    csv_file_names_real = {os.path.basename(x).split(".")[0]: x for x in csv_files_real}
    dataset_names = list(mat_file_names_real.keys()) + list(mat_file_names_syn) + list(csv_file_names_real.keys())
    mat_file_names_real.update(mat_file_names_syn)
    mat_file_names_real.update(csv_file_names_real)
    dataset_paths = mat_file_names_real.copy()

    print("#" * 60)
    print(f"TESTING Model Comparison")
    print("#" * 60)
    print("TEST PARAMETERS:")
    print(f'Models: {args.model_names}')
    print(f"Number of runs: {args.n_runs}")
    print(f"Number of trees: {args.n_trees}")
    print(f"Contamination Factor: {args.contamination}")
    print(f'Distribution point p: {args.distribution}')
    print(f'eta: {args.eta}')
    print(f'Scaler for data normalization: {args.scaler}')
    print(f"Seed: {args.seed}")
    print("#" * 60)

    if len(args.dataset_names) > 0:
        dataset_names = args.dataset_names
    else:
        dataset_names = sorted(dataset_names)

    print("dataset_names", dataset_names)

    print("#" * 60)
    print('Define model:')
    print("#" * 60)
    print(f'Model name: {args.model_names}')
    model=get_model(args.model_names)
    set_p_distribution(model,args.distribution)
    set_p_eta(model,args.eta)
    print(f'Contamination: {model.contamination_}')
    print(f'Distribution: {model.distribution_}')
    print(f'Eta: {model.eta_}')

    print('Define Scaler')
    scaler=get_scaler(args.scaler)
    print(scaler)

    # for name in dataset_names:
    #     print("#" * 60)
    #     print(f"DATASET: {name}")
    #     print("#" * 60)
    #     X_train,X_test,X,y=load_preprocess(name,dataset_paths[name])


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







