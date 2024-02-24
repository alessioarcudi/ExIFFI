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


from model_reboot.EIF_reboot import ExtendedIsolationForest, IsolationForest
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Global Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')

parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=40, help='Global feature importances parameter: n_runs')

parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Interpretation method to use: EXIFFI, DIFFI, RandomForest')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
# Parse the arguments
args = parser.parse_args()

assert args.model in ["IF", "EIF", "EIF+"], "Model not recognized"
assert args.interpretation in ["EXIFFI", "DIFFI", "RandomForest"], "Interpretation not recognized"
if args.interpretation == "DIFFI":
    assert args.model=="IF", "DIFFI can only be used with the IF model"

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path

n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
n_runs = args.n_runs

pre_process = args.pre_process
model = args.model
interpretation = args.interpretation
scenario = args.scenario

# print(f"Dataset: {dataset_name}")
# print(f"Path: {dataset_path}")
# quit()

dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()


if scenario==2:
    #dataset.split_dataset(train_size=0.8,contamination=0)
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

if pre_process:
    dataset.pre_process()

if model == "IF":
    if interpretation == "EXIFFI":
        I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
    elif interpretation == "DIFFI":
        I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF":
    I=ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)

#cwd = '/Users/alessio/Documents/ExIFFI'
cwd = '/home/davidefrizzo/Desktop/PHD/ExIFFI'

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots_new"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

#----------------- GLOBAL IMPORTANCES -----------------#
# initialize global_importances paths
path_experiment = path_experiments + "/global_importances"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
    

path_experiment_model = path_experiment + "/" + model
if not os.path.exists(path_experiment_model):
    os.makedirs(path_experiment_model)
    
path_experiment_model_interpretation = path_experiment_model + "/" + interpretation
if not os.path.exists(path_experiment_model_interpretation):
    os.makedirs(path_experiment_model_interpretation)
path_experiment_model_interpretation_scenario = path_experiment_model_interpretation + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_scenario):
    os.makedirs(path_experiment_model_interpretation_scenario)
    
#Compute global importances
full_importances = experiment_global_importances(I, dataset, n_runs=n_runs, p=contamination, interpretation=interpretation)    
save_element(full_importances, path_experiment_model_interpretation_scenario, filetype="npz")

# plot global importances
most_recent_file = get_most_recent_file(path_experiment_model_interpretation_scenario)
bar_plot(dataset, most_recent_file, filetype="npz", plot_path=path_plots, f=min(dataset.shape[1],6),show_plot=False, model=model, interpretation=interpretation, scenario=scenario)
score_plot(dataset, most_recent_file, plot_path=path_plots, show_plot=False, model=model, interpretation=interpretation, scenario=scenario)

