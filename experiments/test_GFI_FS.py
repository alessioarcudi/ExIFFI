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


from model_reboot.EIF_reboot import ExtendedIsolationForest, IsolationForest
#from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
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
parser.add_argument('--rotation',action='store_true', help='If set, rotate the xticks labels by 45 degrees in the feature selection plot (for ionosphere)')
parser.add_argument('--include_random',action='store_true', help='If set, shows also the random precisions in the feature selection plot')
parser.add_argument('--downsample',action='store_true', help='If set, apply downsample to the big datasets')

# Parse the arguments
args = parser.parse_args()

assert args.model in ["IF", "EIF", "EIF+"], "Model not recognized"
assert args.interpretation in ["EXIFFI+","EXIFFI", "DIFFI", "RandomForest"], "Interpretation not recognized"

if args.interpretation == "DIFFI":
    assert args.model=="IF", "DIFFI can only be used with the IF model"

if args.interpretation == "EXIFFI":
    assert args.model=="EIF", "EXIFFI can only be used with the EIF model"

if args.interpretation == "EXIFFI+":
    assert args.model=="EIF+", "EXIFFI can only be used with the EIF+ model"

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
rotation = args.rotation
include_random = args.include_random
downsample = args.downsample

# Load the dataset
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

if model == "IF":
    if interpretation == "EXIFFI":
        I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
    elif interpretation == "DIFFI" or interpretation == "RandomForest":
        I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF":
    I=ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)

#cwd = '/Users/alessio/Documents/ExIFFI'
cwd = '/home/davidefrizzo/Desktop/PHD/ExIFFI'

print('#'*50)
print('GFI and Feature Selection Experiment')
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
path_plots_imp = cwd +"/experiments/results/"+dataset.name+"/plots_new/imp_plots"
if not os.path.exists(path_plots_imp):
    os.makedirs(path_plots_imp)

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

# Set up the Feature Selection path
    
path_experiment_fs = path_experiments + "/feature_selection"
if not os.path.exists(path_experiment_fs):
    os.makedirs(path_experiment_fs)

path_experiment_model_fs = path_experiment_fs + "/" + model
if not os.path.exists(path_experiment_model_fs):
    os.makedirs(path_experiment_model_fs)
    
path_experiment_model_interpretation_fs = path_experiment_model_fs + "/" + interpretation
if not os.path.exists(path_experiment_model_interpretation_fs):
    os.makedirs(path_experiment_model_interpretation_fs)
path_experiment_model_interpretation_fs_random = path_experiment_model_fs + "/random"
if not os.path.exists(path_experiment_model_interpretation_fs_random):
    os.makedirs(path_experiment_model_interpretation_fs_random)
path_experiment_model_interpretation_scenario_fs = path_experiment_model_interpretation_fs + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_scenario_fs):
    os.makedirs(path_experiment_model_interpretation_scenario_fs)
    
#Compute global importances
full_importances = experiment_global_importances(I, dataset, n_runs=n_runs, p=contamination, interpretation=interpretation)    
save_element(full_importances, path_experiment_model_interpretation_scenario, filetype="npz")

# plot global importances
most_recent_file = get_most_recent_file(path_experiment_model_interpretation_scenario)
bar_plot(dataset, most_recent_file, filetype="npz", plot_path=path_plots_imp, f=min(dataset.shape[1],6),show_plot=False, model=model, interpretation=interpretation, scenario=scenario)
plt.close()
score_plot(dataset, most_recent_file, plot_path=path_plots_imp, show_plot=False, model=model, interpretation=interpretation, scenario=scenario)
plt.close()

# feature selection → direct and inverse feature selection
feat_order = np.argsort(full_importances.mean(axis=0))
Precisions = namedtuple("Precisions",["direct","inverse","dataset","model","value"])
direct = feature_selection(I, dataset, feat_order, 10, inverse=False, random=False)
inverse = feature_selection(I, dataset, feat_order, 10, inverse=True, random=False)
value = abs(sum(direct.mean(axis=1)-inverse.mean(axis=1)))
data = Precisions(direct, inverse, dataset.name, model, value)
save_fs_prec(data, path_experiment_model_interpretation_scenario_fs)

# random feature selection
Precisions_random = namedtuple("Precisions_random",["random","dataset","model"])
random = feature_selection(I, dataset, feat_order, 10, inverse=True, random=True)
data_random = Precisions_random(random, dataset.name, model)
save_fs_prec_random(data_random, path_experiment_model_interpretation_fs_random)

path_plots_fs = cwd +"/experiments/results/"+dataset.name+"/plots_new/fs_plots"
if not os.path.exists(path_plots_fs):
    os.makedirs(path_plots_fs)

#plot feature selection
fs_prec = get_most_recent_file(path_experiment_model_interpretation_scenario_fs)
if include_random:
    fs_prec_random = get_most_recent_file(path_experiment_model_interpretation_fs_random)
    plot_feature_selection(fs_prec, path_plots_fs, fs_prec_random, model=model, interpretation=interpretation, scenario=scenario, plot_image=False,rotation=rotation)
else:
    plot_feature_selection(fs_prec, path_plots_fs, model=model, interpretation=interpretation, scenario=scenario, plot_image=False,rotation=rotation)

