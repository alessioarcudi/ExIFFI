
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
from pyod.models.dif import DIF

import warnings
warnings.simplefilter(action='ignore')

from model_reboot.EIF_reboot import ExtendedIsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Global Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--model', type=str, default="EIF+", help='Name of the model')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Name of the interpretation algorithm')
parser.add_argument('--n_estimators', type=int, default=200, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=npt.NDArray, default=np.linspace(0.01,0.1,10), help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=10, help='Global feature importances parameter: n_runs')
parser.add_argument('--train_size', type=float, default=0.9, help='Global feature importances parameter: train_size')
parser.add_argument('--compute_global_importances', type=bool, default=True, help='Global feature importances parameter: compute_global_importances')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
model = args.model
interpretation = args.interpretation
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
n_runs = args.n_runs
train_size = args.train_size
compute_global_importances = args.compute_global_importances
scenario = args.scenario
pre_process = args.pre_process


isdiffi = model == "DIFFI"
if model not in ["EIF+","EIF","DIFFI"] and compute_global_importances:
    compute_global_importances = False
    print("impossible to compute importances for model ", model)


dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

if scenario==2:
    #dataset.split_dataset(train_size=0.8,contamination=0)
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

if pre_process:
    dataset.pre_process()

if model == "EIF+":
    I = ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF":
    I = ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "IF":
    I = IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "DIF":
    I = DIF(n_estimators=n_estimators, max_samples=max_samples)
    

#cwd = '/Users/alessio/Documents/ExIFFI'
cwd = '/home/davidefrizzo/Desktop/PHD/ExIFFI'

print('#'*50)
print('Precision and Contamination Experiments')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Interpretation Model: {interpretation}')
print(f'Scenario: {scenario}')
print('#'*50)

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots_new"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

#----------------- EVALUATE PRECISIONS OVER CONTAMINATION -----------------#
# initialize contamination paths
path_experiment = path_experiments + "/contamination"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
path_experiment_matrices = path_experiment + "/precisions"
if not os.path.exists(path_experiment_matrices):
    os.makedirs(path_experiment_matrices)

path_importances = path_experiments + "/global_importances/" + "/" + model + "/" + interpretation
if not os.path.exists(path_importances):
    os.makedirs(path_importances)
path_importances_over_contamination = path_importances + "/contamination"
if not os.path.exists(path_importances_over_contamination):
    os.makedirs(path_importances_over_contamination)

#if isdiffi:
# else:
#     path_importances = path_experiment + "/global_importances/EXIFFI"
#     if not os.path.exists(path_importances):
#         os.makedirs(path_importances)
#     path_importances_over_contamination = path_importances + "/contamination"
#     if not os.path.exists(path_importances_over_contamination):
#         os.makedirs(path_importances_over_contamination)

# contamination evaluation
if compute_global_importances:
    precisions, importances = contamination_in_training_precision_evaluation(I, dataset, n_runs, train_size=train_size, contamination_values=contamination, compute_global_importances=compute_global_importances, isdiffi=isdiffi)
    save_element((importances,contamination), path_importances_over_contamination, filetype="pickle")
    save_element((precisions,contamination), path_experiment_matrices, filetype="pickle")
else:
    precisions = contamination_in_training_precision_evaluation(I, dataset, n_runs, train_size=train_size, contamination_values=contamination, compute_global_importances=compute_global_importances, isdiffi=isdiffi)
    save_element((precisions,contamination), path_experiment_matrices, filetype="pickle")

#plot contamination evaluation
(precisions,contamination) = open_element(get_most_recent_file(path_experiment_matrices))
plot_precision_over_contamination(precisions, dataset.name, I.name, interpretation, scenario, path_plots, contamination=contamination, plot_image=False)

