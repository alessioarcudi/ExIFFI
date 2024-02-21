# initialize feature_selection paths
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
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=10, help='Global feature importances parameter: n_runs')

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

path_experiment = path_experiments + "/feature_selection"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
path_feat_sel_values = path_experiment + "/values"
if not os.path.exists(path_feat_sel_values):
    os.makedirs(path_feat_sel_values)
path_experiment_matrices = path_experiment + "/matrices"
if not os.path.exists(path_experiment_matrices):
    os.makedirs(path_experiment_matrices)
    
path_experiment_feats = path_experiments + "/global_importances/EXIFFI/matrices"
if not os.path.exists(path_experiment_feats):
    os.makedirs(path_experiment_feats)
    

# feature selection
most_recent_file = get_most_recent_file(path_experiment_feats)
matrix = open_element(most_recent_file,filetype="npz")
feat_order = np.argsort(matrix.mean(axis=0))
Precisions = namedtuple("Precisions",["direct","inverse","dataset","model"])
direct = feature_selection(I, dataset, feat_order, 10, inverse=False, random=False)
inverse = feature_selection(I, dataset, feat_order, 10, inverse=True, random=False)
data = Precisions(direct, inverse, dataset.name, I.name)
value = abs(sum(direct.mean(axis=1)-inverse.mean(axis=1)))
save_element([data], path_experiment_matrices, filetype="pickle")
save_element(value, path_feat_sel_values, filetype="npz")

#plot feature selection
most_recent_file = get_most_recent_file(path_experiment_matrices)
plot_feature_selection(most_recent_file, path_plots, plot_image=False)