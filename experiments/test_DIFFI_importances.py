import sys
import os
os.chdir('/home/davidefrizzo/PHD/ExIFFI/experiments')
#os.chdir('/Users/alessio/Documents/ExIFFI/experiments')
sys.path.append("..")
from collections import namedtuple

from utils_reboot.experiments import *
from utils_reboot.datasets import *
from utils_reboot.plots import *
from utils_reboot.utils import *

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Global Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=10, help='Global feature importances parameter: n_runs')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
n_estimators = args.n_estimators
max_samples = args.max_samples
contamination = args.contamination
n_runs = args.n_runs

dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

I=IsolationForest(n_estimators=n_estimators,max_samples=max_samples,contamination=args.contamination)

cwd='/home/davidefrizzo/PHD/ExIFFI/experiments'
#cwd = '/Users/alessio/Documents/ExIFFI'

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

#----------------- GLOBAL IMPORTANCES -----------------#
# initialize global_importances paths
path_experiment = path_experiments + "/global_importances"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
path_experiment = path_experiments + "/DIFFI"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
path_experiment_matrices = path_experiment + "/matrices"
if not os.path.exists(path_experiment_matrices):
    os.makedirs(path_experiment_matrices)
path_experiment_plots = path_experiment + "/data_for_plots"
if not os.path.exists(path_experiment_plots):
    os.makedirs(path_experiment_plots)

#Compute global importances
full_importances, data_for_plots = compute_global_importances(I, dataset,n_runs,p=contamination,isdiffi=True)    
save_element(full_importances, path_experiment_matrices, filetype="npz")
save_element(data_for_plots, path_experiment_plots, filetype="pickle")

# plot global importances
most_recent_file = get_most_recent_file(path_experiment_matrices)
bar_plot(dataset, most_recent_file, filetype="npz", plot_path=path_plots, show_plot=False)
most_recent_file = get_most_recent_file(path_experiment_plots)
score_plot(dataset, most_recent_file, plot_path=path_plots, show_plot=False)

#----------------- LOCAL SCOREMAP -----------------#
# initialize local_scoremap paths
path_experiment = path_experiments + "/local_scoremap"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)

# Compute local scoremap
I.fit(dataset.X)
importance_map(dataset,I,feats_plot=(12,4),path_plot=path_plots,isdiffi=True,iforest=I)