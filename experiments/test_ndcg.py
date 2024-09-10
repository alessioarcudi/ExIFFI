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
from sklearn.metrics import ndcg_score

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='NDCG Experiment')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
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
model = args.model
interpretation = args.interpretation
scenario = args.scenario

# Load the dataset
dataset = Dataset(dataset_name, path = dataset_path,feature_names_filepath='../data/')
dataset.drop_duplicates()

os.chdir('../')
cwd=os.getcwd()

print('#'*50)
print('NDCG Experiment')
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

path_experiment = path_experiments + "/global_importances"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)

path_ndcg = path_experiments + "/ndcg"
if not os.path.exists(path_ndcg):
    os.makedirs(path_ndcg)
    
path_experiment_model = path_experiment + "/" + model
if not os.path.exists(path_experiment_model):
    os.makedirs(path_experiment_model)

path_ndcg_model = path_ndcg + "/" + model
if not os.path.exists(path_ndcg_model):
    os.makedirs(path_ndcg_model)
    
path_experiment_model_interpretation = path_experiment_model + "/" + interpretation
if not os.path.exists(path_experiment_model_interpretation):
    os.makedirs(path_experiment_model_interpretation)

path_ndcg_model_interpretation = path_ndcg_model + "/" + interpretation
if not os.path.exists(path_ndcg_model_interpretation):
    os.makedirs(path_ndcg_model_interpretation)

path_experiment_model_interpretation_scenario = path_experiment_model_interpretation + "/scenario_"+str(scenario)
if not os.path.exists(path_experiment_model_interpretation_scenario):
    os.makedirs(path_experiment_model_interpretation_scenario)

path_ndcg_model_interpretation_scenario = path_ndcg_model_interpretation + "/scenario_"+str(scenario)
if not os.path.exists(path_ndcg_model_interpretation_scenario):
    os.makedirs(path_ndcg_model_interpretation_scenario)

# Use path_experiment_model_interpretation_scenario to retrieve the global importances matrix
imp_data_path=get_most_recent_file(path_experiment_model_interpretation_scenario,filetype="npz")
imp_mat=open_element(imp_data_path,filetype="npz")
# Compute the mean importances over the different runs, this will be the importance scores
# to compare with the true_relevance feature ranking to compute the NDCG score
importance_scores=[np.mean(imp_mat,axis=0)]

# Get the true_relevance feature ranking from true_relevance_dict.pickle
true_relevance_dict_path=os.path.join(cwd,'utils_reboot','true_relevance_dict.pickle')
true_relevance_dict=open_element(true_relevance_dict_path)
true_relevance=[true_relevance_dict[dataset_name]]

# import ipdb; ipdb.set_trace()

# Compute the NDCG score
ndcg = ndcg_score(true_relevance, importance_scores)
print('#'*50)
print(f'NDCG score for {dataset.name} with {model}_{interpretation} in scenario {str(scenario)}: {ndcg}')
print('#'*50)

# Save the NDCG score
ndcg_dict={'NDCG':ndcg}
save_element(ndcg_dict, path_ndcg_model_interpretation_scenario, filetype="pickle")





