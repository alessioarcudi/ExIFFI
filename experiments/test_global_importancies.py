import sys
import os
#os.chdir('/Users/alessio/Documents/ExIFFI/experiments')
os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
sys.path.append("..")
from collections import namedtuple

from utils_reboot.experiments import *
from utils_reboot.datasets import *
from utils_reboot.plots import *
from utils_reboot.utils import *


from model_reboot.EIF_reboot import ExtendedIsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Global Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--plus', type=bool, default=True, help='EIF parameter: plus')
parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.08, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=10, help='Global feature importances parameter: n_runs')
parser.add_argument('--model', type=str, default="EXIFFI", help='Global feature importances parameter: model_index')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
plus = args.plus
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
n_runs = args.n_runs
model = args.model


dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()
#dataset.pre_process()

I=ExtendedIsolationForest(plus, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)

#cwd = '/Users/alessio/Documents/ExIFFI'
cwd = '/home/davidefrizzo/Desktop/PHD/ExIFFI'

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
    
if model == "EXIFFI":
    path_experiment_model = path_experiment + "/EXIFFI"
    if not os.path.exists(path_experiment_model):
        os.makedirs(path_experiment_model)
elif model == "DIFFI":
    path_experiment_model = path_experiment + "/DIFFI"
    if not os.path.exists(path_experiment_model):
        os.makedirs(path_experiment_model)
    
path_experiment_model_matrices = path_experiment_model + "/matrices"
if not os.path.exists(path_experiment_model_matrices):
    os.makedirs(path_experiment_model_matrices)
path_experiment_model_data_for_plots = path_experiment + "/data_for_plots"
if not os.path.exists(path_experiment_model_data_for_plots):
    os.makedirs(path_experiment_model_data_for_plots)
    
#Compute global importances
full_importances = experiment_global_importances(I, dataset, n_runs=n_runs, p=contamination)    
save_element(full_importances, path_experiment_model_matrices, filetype="npz")

# plot global importances
print(path_experiment_model_matrices)
most_recent_file = get_most_recent_file(path_experiment_model_matrices)
bar_plot(dataset, most_recent_file, filetype="npz", plot_path=path_plots, show_plot=False, model_name="EIF"+plus*"+")
score_plot(dataset, most_recent_file, plot_path=path_plots, show_plot=False, model_name="EIF"+plus*"+")

