# initialize feature_selection paths
import sys
import ast
import os
cwd = os.getcwd()
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
parser = argparse.ArgumentParser(description='Test Feature Selection')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--plus', type=bool, default=True, help='EIF parameter: plus')
parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=10, help='Global feature importances parameter: n_runs')
parser.add_argument('--model', type=str, default="EIF+", help='Name of the AD model. Accepted values are: [IF,EIF,EIF+,DIF,AE]')
parser.add_argument('--model_interpretation',type=str, default="EIF+", help='Name of the model from which we take feature order for the Feature Selection plot')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Name of the interpretation model. Accepted values are: [EXIFFI,DIFFI,RF,TreeSHAP]')
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument('--split',action='store_true', help='If set, split the dataset when pre procesing')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--rotation',action='store_true', help='If set, rotate the xticks labels by 45 degrees in the feature selection plot (for ionosphere)')
parser.add_argument('--compute_random',action='store_true', help='If set, shows also the random precisions in the feature selection plot')

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
model_interpretation = args.model_interpretation
interpretation = args.interpretation
pre_process = args.pre_process
split = args.split
scenario = args.scenario
rotation = args.rotation
compute_random = args.compute_random


dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

if scenario==2:
    dataset.split_dataset(train_size=0.8,contamination=0)

if pre_process:
    dataset.pre_process()

assert model_interpretation in ["IF", "EIF", "EIF+"], "Model for Feature Order not recognized"
assert model in ["EIF", "EIF+"], "Evaluation Model not recognized"
assert interpretation in ["EXIFFI+","EXIFFI", "DIFFI", "RandomForest"], "Interpretation not recognized"

if interpretation == "DIFFI":
    assert model_interpretation=="IF", "DIFFI can only be used with the IF model"

if interpretation == "EXIFFI":
    assert model_interpretation=="EIF", "EXIFFI can only be used with the EIF model"

if interpretation == "EXIFFI+":
    assert model_interpretation=="EIF+", "EXIFFI+ can only be used with the EIF+ model"

if model == "IF":
    if interpretation == "EXIFFI":
        I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
    elif interpretation == "DIFFI" or interpretation == "RandomForest":
        I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF":
    I=ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)


print('#'*50)
print('Feature Selection Experiment')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Model for Feature Order: {model_interpretation}')
print(f'Interpretation Model: {interpretation}')
print(f'Scenario: {scenario}')
print('#'*50)

cwd = '/home/davidefrizzo/Desktop/PHD/ExIFFI'

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots_new/fs_plots"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

path_experiment = path_experiments + "/feature_selection"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)

path_experiment_model = path_experiment + "/" + model
if not os.path.exists(path_experiment_model):
    os.makedirs(path_experiment_model)
    
path_experiment_model_interpretation = path_experiment_model + "/" + model_interpretation + "_" + interpretation
if not os.path.exists(path_experiment_model_interpretation):
    os.makedirs(path_experiment_model_interpretation)
path_experiment_model_interpretation_random = path_experiment_model + "/random" 
if not os.path.exists(path_experiment_model_interpretation_random):
    os.makedirs(path_experiment_model_interpretation_random)

path_experiment_model_interpretation_scenario = path_experiment_model_interpretation + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_scenario):
    os.makedirs(path_experiment_model_interpretation_scenario)
path_experiment_model_interpretation_random_scenario = path_experiment_model_interpretation_random + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_random_scenario):
    os.makedirs(path_experiment_model_interpretation_random_scenario)
    
path_experiment_feats = path_experiments + "/global_importances/" + model_interpretation + "/" + interpretation + "/scenario_" + str(scenario)
if not os.path.exists(path_experiment_feats):
    os.makedirs(path_experiment_feats) 

# feature selection → direct and inverse feature selection
most_recent_file = get_most_recent_file(path_experiment_feats)
matrix = open_element(most_recent_file,filetype="npz")
feat_order = np.argsort(matrix.mean(axis=0))
Precisions = namedtuple("Precisions",["direct","inverse","dataset","model","value"])
direct = feature_selection(I, dataset, feat_order, 10, inverse=False, random=False)
inverse = feature_selection(I, dataset, feat_order, 10, inverse=True, random=False)
value = abs(sum(direct.mean(axis=1)-inverse.mean(axis=1)))
data = Precisions(direct, inverse, dataset.name, model, value)
save_fs_prec(data, path_experiment_model_interpretation_scenario)

# random feature selection
if compute_random:
    Precisions_random = namedtuple("Precisions_random",["random","dataset","model"])
    random = feature_selection(I, dataset, feat_order, 10, inverse=True, random=True)
    data_random = Precisions_random(random, dataset.name, model)
    save_fs_prec_random(data_random, path_experiment_model_interpretation_random_scenario)

#plot feature selection
fs_prec = get_most_recent_file(path_experiment_model_interpretation_scenario)
fs_prec_random = get_most_recent_file(path_experiment_model_interpretation_random_scenario)
plot_feature_selection(fs_prec, path_plots, fs_prec_random, model=model_interpretation, eval_model=model, interpretation=interpretation, scenario=scenario, plot_image=False,rotation=rotation)


