"""
Python Script to launch the ExIFFI experiments.
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
from scipy.io import loadmat

from test_model import get_filename

sys.path.append('../src')
from src.utils import *

sys.path.append('../models')
from models.Extended_DIFFI_original import *
from models.Extended_DIFFI_parallel import * 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def parse_arguments():
    parser=argparse.ArgumentParser()

    # Set the contaminatin factor in the training set
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
        default=100,
        help="Number of trees in the forest"
    )

    # Include or exclude the depth_based parameter
    parser.add_argument(
        "--depth_based",
        type=bool,
        default=False,
        help="Include depth_based parameter"
    )

    # save directory for the results
    parser.add_argument(
        "--savedir",
        type=str,
        required=True,
        help="Save directory for the results"
    )

    # Set the number of runs for the GFI Computation
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of runs for the GFI computation"
    )

    # List of datasets to be used
    parser.add_argument(
        "--dataset_names",
        required=True,
        nargs="+",
        type=str,
        help="List of names of datasets on which to run the experiments"
    )

    # Set the seed 
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set seed for reproducibility"
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

    # Set the number of cores to use
    parser.add_argument(
    "--n_cores",
    type=int,
    nargs="+",
    default=[1],
    help="Set number of cores to use. "
    + "If [1] the code is serial, otherwise it is parallel. "
    + "List of 1 or 3 integers, respectively num processes of fit, importance and anomaly",
    )

    # Run the wrapper for timing the code
    parser.add_argument(
    "--wrapper",
    action="store_true",
    help="If set, run the wrapper for timing the code",
    )

    # add_bash -c to the command for timing the code
    parser.add_argument(
        "--add_bash",
        action="store_true",
        help="Include bash script"
    )

    parser.add_argument(
    "--filename",
    type=str,   
    default=None,
    help="Filename of the output saved file. If None, it is automatically generated",
    
    )
    return parser.parse_args()

def test_exiffi(
X_train,
X_test,
y_test,
savedir: str,
n_cores_fit,
n_cores_importance,
n_cores_anomaly,
distribution,
n_runs=10,
depth_based=False,
seed=None,
n_trees=300,
name="",
filename=None,
):
    args_to_avoid = ["X_train", "X_test", "savedir", "args_to_avoid", "args"]
    args = dict()
    for k, v in locals().items():
        if k in args_to_avoid:
            continue
        args[k] = v

    if seed is not None:
        np.random.seed(seed)

    if filename is None:
        filename = get_filename(name,partial_name="exiffi_test")

    plt_data_path = os.path.join(savedir, "plt_data")
    imp_path = os.path.join(savedir, "imps")
    plot_path = os.path.join(savedir, "plots")
    npz_path = os.path.join(savedir,"npz")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    if not os.path.exists(imp_path):
        os.makedirs(imp_path)
    if not os.path.exists(plt_data_path):
        os.makedirs(plt_data_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    filepath = os.path.join(npz_path, filename)

    EDIFFI = Extended_DIFFI_parallel(
        n_trees=n_trees, max_depth=100, subsample_size=256, plus=1
    )
    EDIFFI.set_num_processes(n_cores_fit, n_cores_importance, n_cores_anomaly)
    set_p_distribution(EDIFFI,distribution)

    start=time.time()
    # Compute the global importances
    print('Computing Global Importances')
    fi,plt_data,fi_path,plt_data_path=EDIFFI.compute_global_importances(X_test,
                                                                        n_runs,
                                                                        name,
                                                                        True,
                                                                        True,
                                                                        depth_based,
                                                                        imp_path,
                                                                        plt_data_path
                                                                        )
    
    # Produce the Bar Plot
    print('Producing Bar Plot')
    fig,ax,bars=EDIFFI.bar_plot(fi_path,name,plot_path,show_plot=False)
    print(f'Bar Plot saved in {plot_path}')

    # Produce the Score Plot 
    print('Producing Score Plot')
    ax1,ax2=EDIFFI.score_plot(plt_data_path,name,plot_path,show_plot=False)
    print(f'Score Plot saved in {plot_path}')

    # Produce the Importance Scoremap
    print('Producing Importance Scoremap')
    fig,ax=EDIFFI.importance_map(name,X_test,y_test,30,plot_path,show_plot=False)
    print(f'Importance Map saved in {plot_path}')

    # Produce the Complete Scoremap
    print('Producing Complete Scoremap')
    fig,ax=EDIFFI.complete_scoremap(name,X_test.shape[1],X_test,y_test,plot_path,show_plot=False,half=True)
    print(f'Complete Scoremap saved in {plot_path}')

    end=time.time()
    time_stat = end-start

    np.savez(
    filepath,
    execution_time_stat=time_stat,
    bars=bars,
    global_imp=fi,
    arguments=args,
    time=pd.Timestamp.now(),
    )
    print(f"Results saved in {filepath}")

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

    if len(args.n_cores) == 1:
        n_cores_fit = args.n_cores[0]
        n_cores_importance = args.n_cores[0]
        n_cores_anomaly = args.n_cores[0]
    elif len(args.n_cores) == 3:
        n_cores_fit = args.n_cores[0]
        n_cores_importance = args.n_cores[1]
        n_cores_anomaly = args.n_cores[2]
    else:
        raise ValueError("Number of elements in --n_cores must be either 1 or 3")

    print("#" * 60)
    print(f"TESTING ExIFFI")
    print("#" * 60)
    print("TEST PARAMETERS:")
    print(f"Number of runs: {args.n_runs}")
    print(f"Number of trees: {args.n_trees}")
    print(f"Depth based: {args.depth_based}")
    print(f"Contamination Factor: {args.contamination}")
    print(f'Distribution point p: {args.distribution}')
    print(f'Scaler for data normalization: {args.scaler}')
    print(
        f"Number of cores: fit {n_cores_fit}, importance {n_cores_importance}, anomaly {n_cores_anomaly}"
    )
    print(f"Seed: {args.seed}")
    print("#" * 60)
    
    if len(args.dataset_names) > 0:
        dataset_names = args.dataset_names
    else:
        dataset_names = sorted(dataset_names)

    print("dataset_names", dataset_names)

    for name in dataset_names:
        print("#" * 60)
        print(f"DATASET: {name}")
        print("#" * 60)
        X_train,X_test,X,y=load_preprocess(name,dataset_paths[name])
        test_exiffi(
            X_train=X_train,
            X_test=X_test,
            y_test=y,
            savedir=args.savedir,
            n_cores_fit=n_cores_fit,
            n_cores_importance=n_cores_importance,
            n_cores_anomaly=n_cores_anomaly,
            distribution=args.distribution,
            n_runs=args.n_runs,
            depth_based=args.depth_based,
            seed=args.seed,
            n_trees=args.n_trees,
            name=name,
            filename=args.filename,
        )

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

  