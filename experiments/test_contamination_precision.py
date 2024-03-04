
import sys
import os
cwd = os.getcwd()
os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
sys.path.append("..")
from collections import namedtuple

import warnings
warnings.simplefilter(action='ignore')

from utils_reboot.experiments import *
from utils_reboot.datasets import *
from utils_reboot.plots import *
from utils_reboot.utils import *
#from pyod.models.dif import DIF
from pyod.models.auto_encoder import AutoEncoder
#from sklearn.ensemble import IsolationForest as sklearn_IsolationForest

from model_reboot.EIF_reboot import ExtendedIsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Global Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=200, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=npt.NDArray, default=np.linspace(0.0,0.1,10), help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=10, help='Global feature importances parameter: n_runs')
parser.add_argument('--train_size', type=float, default=0.9, help='Global feature importances parameter: train_size')
parser.add_argument('--compute_GFI', type=bool, default=False, help='Global feature importances parameter: compute_GFI')
parser.add_argument('--downsample',action='store_true', help='If set, apply downsample to the big datasets')
parser.add_argument('--model', type=str, default="DIF", help='Name of the model')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Name of the interpretation algorithm')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
n_runs = args.n_runs
train_size = args.train_size
downsample = args.downsample
GFI = args.compute_GFI
model = args.model
interpretation = args.interpretation
scenario = args.scenario
pre_process = args.pre_process

assert model in ["IF", "EIF", "EIF+", "DIF", "AnomalyAutoencoder"], "Model not recognized"
assert interpretation in ["EXIFFI+", "EXIFFI", "DIFFI", "RandomForest"], "Interpretation not recognized"

if interpretation == "DIFFI":
    assert model=="IF", "DIFFI can only be used with the IF model"

if interpretation == "EXIFFI":
    assert model=="EIF", "EXIFFI can only be used with the EIF model"

if interpretation == "EXIFFI+":
    assert model=="EIF+", "EXIFFI+ can only be used with the EIF+ model"


if model == "DIF" and GFI:
    raise ValueError("DIF model does not support global feature importances")
if model == "AnomalyAutoencoder" and GFI:
    raise ValueError("AnomalyAutoencoder model does not support global feature importances")

dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

# Downsample datasets with more than 7500 samples (i.e. diabetes shuttle and moodify)
if downsample:
    dataset.downsample(max_samples=7500)

# Set scenario and scale the data
if scenario==2:
    #dataset.split_dataset(train_size=0.8,contamination=0)
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)
if pre_process:
    dataset.pre_process()

if model == "EIF+":
    I = ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF":
    I = ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif (model == "IF" and interpretation == "EXIFFI") or (model =="IF" and not GFI):
    I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "IF" and interpretation == "DIFFI":
    I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "DIF":
    I = DIF(n_estimators=n_estimators, max_samples=max_samples)
elif model == "AnomalyAutoencoder":
    I = AutoEncoder(hidden_neurons=[dataset.X.shape[1], 32, 32, dataset.X.shape[1]], contamination=0.1, epochs=100, random_state=42,verbose =0)
    

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
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots_new/contamination_plots"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

#----------------- EVALUATE PRECISIONS OVER CONTAMINATION -----------------#
# initialize contamiination paths
path_experiment_contamination = path_experiments + "/contamination"
if not os.path.exists(path_experiment_contamination):
    os.makedirs(path_experiment_contamination)
path_experiment_global_importances = path_experiments + "/global_importances/contamination"
if not os.path.exists(path_experiment_global_importances):
    os.makedirs(path_experiment_global_importances)

path_experiment_contamination_model = path_experiment_contamination + "/" + model
if not os.path.exists(path_experiment_contamination_model):
    os.makedirs(path_experiment_contamination_model)

path_experiment_contamination_model_scenario = path_experiment_contamination + "/" + model + "/scenario_" + str(scenario)
if not os.path.exists(path_experiment_contamination_model_scenario):
    os.makedirs(path_experiment_contamination_model_scenario)

path_experiment_global_importances_model = path_experiment_global_importances + "/" + model
if not os.path.exists(path_experiment_global_importances_model):
    os.makedirs(path_experiment_global_importances_model)
path_experiment_global_importances_model_interpretation = path_experiment_global_importances_model + "/" + interpretation
if not os.path.exists(path_experiment_global_importances_model_interpretation):
    os.makedirs(path_experiment_global_importances_model_interpretation)

path_experiment_global_importances_model_interpretation_scenario = path_experiment_global_importances_model + "/" + interpretation + "/scenario_" + str(scenario)
if not os.path.exists(path_experiment_global_importances_model_interpretation_scenario):
    os.makedirs(path_experiment_global_importances_model_interpretation_scenario)

# contamination evaluation
if GFI:
    precisions, importances = contamination_in_training_precision_evaluation(I, dataset, n_runs, train_size=train_size, contamination_values=contamination, compute_GFI=GFI, interpretation = interpretation)
    save_element((importances,contamination), path_experiment_global_importances_model_interpretation_scenario, filetype="pickle")
    save_element((precisions,contamination), path_experiment_contamination_model_scenario, filetype="pickle")
else:
    precisions = contamination_in_training_precision_evaluation(I, dataset, n_runs, train_size=train_size, contamination_values=contamination, compute_GFI=GFI, interpretation = interpretation)
    save_element((precisions,contamination), path_experiment_contamination_model_scenario, filetype="pickle")

#plot contamination evaluation
(precisions,contamination) = open_element(get_most_recent_file(path_experiment_contamination_model_scenario))
plot_precision_over_contamination(precisions, dataset.name, model, interpretation, scenario, path_plots, contamination=contamination, plot_image=False)

