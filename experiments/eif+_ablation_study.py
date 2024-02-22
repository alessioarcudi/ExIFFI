import sys
import os
os.chdir('/Users/alessio/Documents/ExIFFI/experiments')
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
parser.add_argument('--model_index', type=int, default=8, help='Global feature importances parameter: model_index')

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
model_index = args.model_index


dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

I=ExtendedIsolationForest(plus, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)

cwd = '/Users/alessio/Documents/ExIFFI'

path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)
path_plots = cwd +"/experiments/results/"+dataset.name+"/plots"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)
    

# Ablation study of the contamination parameter of the GIF --> the results come from the contamination over precision experiment
path = "/Users/alessio/Documents/ExIFFI/experiments/results/wine/experiments/global_importances/matrices/contamination"
try:
    (importances,contamination) = open_element(get_most_recent_file(path))
except:
    raise ValueError("The file does not exist, you have to run contamination over precision experiment first")

#the importances matrix is a 4D tensor with:
#- the first dimension representing the different contaminations over which the model was trained
#- the second dimension representing the different runs
#- the third dimension representing the different contaminations to evaluate the global importances
#- the fourth dimension representing the different features

# the plot will be the importance of the features over the contamination values  in a single run of the model

gfi_over_contamination(importances,contamination,model_index,path_plots)

