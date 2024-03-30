import sys
import os
cwd = os.getcwd()
sys.path.append("..")
from collections import namedtuple

from utils_reboot.experiments import *
from utils_reboot.datasets import *
from utils_reboot.plots import *
from utils_reboot.utils import *

from model_reboot.EIF_reboot import ExtendedIsolationForest
from model_reboot.EIF_reboot import IsolationForest as EIF_IsolationForest
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Performance Metrics')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='Xaxis', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/syn/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=200, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=40, help='Global feature importances parameter: n_runs')
parser.add_argument('--pre_process',type=bool, default=False, help='If set, preprocess the dataset')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--change_ylim',type=bool,default=False, help='If set, change the ylim of the plot from 1 to 1.1')

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
pre_process = args.pre_process
model = args.model
scenario = args.scenario
change_ylim = args.change_ylim

#import ipdb; ipdb.set_trace()

# Load the dataset
dataset = Dataset(dataset_name, path = dataset_path)
dataset.drop_duplicates()

# Downsample datasets with more than 7500 samples
if dataset.shape[0] > 7500:
    dataset.downsample(max_samples=7500)

if scenario==2:
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

# import ipdb; 
# ipdb.set_trace()

# Preprocess the dataset
if pre_process:
    print("#"*50)
    print("Preprocessing the dataset...")
    print("#"*50)
    dataset.pre_process()
elif scenario==2 and not pre_process:
    print("#"*50)
    print("Dataset not preprocessed")
    #dataset.initialize_train()
    dataset.initialize_test()
    print("#"*50)
elif scenario==1 and not pre_process:
    print("#"*50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#"*50)

os.chdir('../')
cwd=os.getcwd()

assert model in ["IF","sklearn_IF","EIF", "EIF+","DIF","AnomalyAutoencoder"], "Evaluation Model not recognized"

I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)


names = [
    "Xaxis", "bisect", "bisect_3d", "bisect_6d", "annthyroid", "breastw", "cardio", "diabetes", 
    "glass", "ionosphere", "moodify", "pendigits", "pima", "shuttle", "wine"
]

EIF_values = [0.99, 0.99, 0.99, 0.99, 0.45, 0.98, 0.74, 0.55, 0.57, 0.90, 0.65, 0.27, 0.54, 0.97, 0.57]

EIF_dataset_values_dict = dict(zip(names, EIF_values))

eta_list = np.linspace(0.5,5,25)
avg_prec = ablation_EIF_plus(I,dataset,eta_list)


path_ablation = cwd+"/experiments/results/"+dataset.name+"/experiments/ablationEIF+"
if not os.path.exists(path_ablation):
    os.makedirs(path_ablation)

plot_path = cwd+"/experiments/results/"+dataset.name+"/plots_new/ablationEIF+"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

    
#save_element(avg_prec, path_ablation, filetype="pickle")

avg_prec_file = get_most_recent_file(path_ablation)
avg_prec = open_element(avg_prec_file, filetype="pickle")
EIF_value = EIF_dataset_values_dict[dataset_name]

plot_ablation(eta_list,avg_prec,EIF_value,
                        dataset_name,
                        plot_path=plot_path,
                        change_ylim=change_ylim)