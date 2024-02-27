# initialize feature_selection paths
import sys
import ast
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
parser.add_argument('--interpretation', type=str, default="EXIFFI", help='Name of the interpretation model. Accepted values are: [EXIFFI,DIFFI,RF,TreeSHAP]')
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument('--split',action='store_true', help='If set, split the dataset when pre procesing')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--box_loc', type=str, default=(3,0.8), help='Location of the box in the feature selection plot')
parser.add_argument('--rotation',action='store_true', help='If set, rotate the xticks labels by 45 degrees in the feature selection plot (for ionosphere)')

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
interpretation = args.interpretation
pre_process = args.pre_process
split = args.split
scenario = args.scenario
box_loc_str = args.box_loc
box_loc=ast.literal_eval(box_loc_str)
rotation = args.rotation


dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

if scenario==2:
    dataset.split_dataset(train_size=0.8,contamination=0)

if pre_process:
    dataset.pre_process()

if model == "EIF+" or model == "EIF":
    I=ExtendedIsolationForest(plus, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "IF":
    I=IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)
# elif model == "AE":
#     I=AutoEncoder(contamination=contamination)

#cwd = '/Users/alessio/Documents/ExIFFI'
cwd = '/home/davidefrizzo/Desktop/PHD/ExIFFI'

print('#'*50)
print('Feature Selection Experiment')
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

path_experiment = path_experiments + "/feature_selection"
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
    
    
path_experiment_feats = path_experiments + "/global_importances/" + model + "/" + interpretation + "/scenario_" + str(scenario)
if not os.path.exists(path_experiment_feats):
    os.makedirs(path_experiment_feats) 

# feature selection
most_recent_file = get_most_recent_file(path_experiment_feats)
matrix = open_element(most_recent_file,filetype="npz")
feat_order = np.argsort(matrix.mean(axis=0))
Precisions = namedtuple("Precisions",["direct","inverse","dataset","model","value"])
direct = feature_selection(I, dataset, feat_order, 10, inverse=False, random=False)
inverse = feature_selection(I, dataset, feat_order, 10, inverse=True, random=False)
value = sum(direct.mean(axis=1)-inverse.mean(axis=1))
data = Precisions(direct, inverse, dataset.name, I.name, value)
save_element([data], path_experiment_model_interpretation_scenario, filetype="pickle")

#plot feature selection
most_recent_file = get_most_recent_file(path_experiment_model_interpretation_scenario)
plot_feature_selection(most_recent_file, path_plots, model=model, plot_image=False,box_loc=dataset.box_loc)




# if model == "EXIFFI":
#     path_experiment_model = path_experiment + "/" + model + "/interpretation_name"
#     if not os.path.exists(path_experiment_model):
#         os.makedirs(path_experiment_model)
# elif model == "DIFFI":
#     path_experiment_model = path_experiment + "/" + model + "/interpretation_name"
#     if not os.path.exists(path_experiment_model):
#         os.makedirs(path_experiment_model)
# elif model == "RF":
#     path_experiment_model = path_experiment + "/RF"
#     if not os.path.exists(path_experiment_model):
#         os.makedirs(path_experiment_model)
# elif model == "TS":
#     path_experiment_model = path_experiment + "/TS"
#     if not os.path.exists(path_experiment_model):
#         os.makedirs(path_experiment_model)