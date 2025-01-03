import sys
import os
cwd = os.getcwd()
#os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
sys.path.append("..")
from collections import namedtuple

from utils_reboot.experiments import *
from utils_reboot.datasets import *
from utils_reboot.plots import *
from utils_reboot.utils import *


from model_reboot.EIF_reboot import ExtendedIsolationForest, IsolationForest
# from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
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
parser.add_argument('--percentile', type=float, default=0.99, help='Percentile to use in the ECOD GFI computation')
parser.add_argument('--n_runs', type=int, default=40, help='Global feature importances parameter: n_runs')
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+')
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Interpretation method to use: EXIFFI, DIFFI, RandomForest')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
# Parse the arguments
args = parser.parse_args()

assert args.model in ["IF", "EIF", "EIF+","ECOD"], "Model not recognized"
assert args.interpretation in ["EXIFFI+", "EXIFFI", "DIFFI", "RandomForest", "ECOD"], "Interpretation not recognized"

if args.interpretation == "DIFFI":
    assert args.model=="IF", "DIFFI can only be used with the IF model"

if args.interpretation == "ECOD":
    assert args.model=="ECOD", "ECOD can only be used with the ECOD model"

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
percentile = args.percentile 
n_runs = args.n_runs
pre_process = args.pre_process
model = args.model
interpretation = args.interpretation
scenario = args.scenario

# Load the dataset
dataset = Dataset(dataset_name, path = dataset_path,feature_names_filepath='../data/')
dataset.drop_duplicates()

# import ipdb; ipdb.set_trace()

# Downsample datasets with more than 7500 samples
if dataset.shape[0] > 7500:
    dataset.downsample(max_samples=7500)

if scenario==2:
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

# Preprocess the dataset
if pre_process:
    print("#"*50)
    print("Preprocessing the dataset...")
    print("#"*50)
    dataset.pre_process()
else:
    print("#"*50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#"*50)


if model == "IF" and interpretation == "EXIFFI":
        I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "IF" and (interpretation in ["DIFFI","RandomForest"]):
        I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF":
    I=ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF+":
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "ECOD":
    I=ECOD(contamination=contamination)

# import ipdb; ipdb.set_trace()
    
os.chdir('../')
cwd=os.getcwd()

print('#'*50)
print('GFI Experiment')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Interpretation Model: {interpretation}')
print(f'Scenario: {scenario}')
print(f'Percentile: {percentile}')
print('#'*50)

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots_new/imp_plots"
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

# Compute global importances → here we have to put p=contamination because we are 
# in an unsupervised scenario where we do not know the true contamination of the dataset
full_importances,_ = experiment_global_importances(I, dataset, n_runs=n_runs, p=contamination, interpretation=interpretation, percentile=percentile)    
save_element(full_importances, path_experiment_model_interpretation_scenario, filetype="npz")

# plot global importances
most_recent_file = get_most_recent_file(path_experiment_model_interpretation_scenario,filetype="npz")

if n_runs>1:
    bar_plot(dataset, most_recent_file, filetype="npz", plot_path=path_plots, f=min(dataset.shape[1],6),show_plot=False, model=model, interpretation=interpretation, scenario=scenario)

score_plot(dataset, most_recent_file, plot_path=path_plots, show_plot=False, model=model, interpretation=interpretation, scenario=scenario)

