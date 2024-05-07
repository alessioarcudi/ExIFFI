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
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
import argparse

class sklearn_IF(sklearn_IsolationForest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "sklearn_IF"

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Time Scaling')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=100, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=float, default=0.1, help='Global feature importances parameter: contamination')
parser.add_argument('--n_runs', type=int, default=40, help='Global feature importances parameter: n_runs')
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+,EIF+_RF,DIF,AnomalyAutoencoder,AnomalyAutoencoder_16')
parser.add_argument('--interpretation', type=str, default="NA", help='Interpretation method to use: EXIFFI, DIFFI, RandomForest')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--compute_GFI', type=bool, default=False, help='Global feature importances parameter: compute_GFI')
parser.add_argument('--compute_fit_predict', type=bool, default=False, help='Weather to compute fit_predict experiment')

# Parse the arguments
args = parser.parse_args()

assert args.model in ["IF", "EIF", "EIF+","EIF+_RF","sklearn_IF","DIF","AnomalyAutoencoder","AnomalyAutoencoder_16"], "Model not recognized"
assert args.interpretation in ["EXIFFI+", "EXIFFI", "DIFFI", "RandomForest","NA"], "Interpretation not recognized"

if args.interpretation == "EXIFFI+":
    assert args.model=="EIF+", "EXIFFI+ can only be used with the EIF+ model"

if args.interpretation == "EXIFFI":
    assert args.model=="EIF" or args.model=="IF", "EXIFFI can only be used with the IF and EIF model"

if args.interpretation == "DIFFI":
    assert args.model=="IF", "DIFFI can only be used with the IF model"

if args.interpretation == "RandomForest":
    assert args.model=='EIF+_RF', "For the time scaling experiments only EIF+_RF model is accepted for the RandomForest interpretation"

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
GFI = args.compute_GFI
fit_predict = args.compute_fit_predict 

dataset = Dataset(dataset_name, path = dataset_path,feature_names_filepath='../data/')
dataset.drop_duplicates()

if dataset.shape[0]>7500:
    dataset.downsample(max_samples=7500)

if scenario==2:
    #dataset.split_dataset(train_size=0.8,contamination=0)
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

if model == "IF":
    if interpretation == "EXIFFI":
        I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
    elif interpretation == "DIFFI" or interpretation == "RandomForest" or model=="sklearn_IF":
        I = sklearn_IF(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF":
    I=ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF+" or model=='EIF+_RF':
    I=ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "DIF":
    I = DIF(max_samples=max_samples)
elif model == "AnomalyAutoencoder":
    I = AutoEncoder(hidden_neurons=[dataset.X.shape[1], 32, 32, dataset.X.shape[1]], contamination=0.1, epochs=50, random_state=42,verbose=0)
elif model == "AnomalyAutoencoder_16":
    I = AutoEncoder(hidden_neurons=[dataset.X.shape[1], 16, 16, dataset.X.shape[1]], contamination=0.1, epochs=50, random_state=42,verbose=0)

if model == "DIF" and GFI:
    raise ValueError("DIF model does not support global feature importances")
if (model == "AnomalyAutoencoder" or model == "AnomalyAutoencoder_16")  and GFI:
    raise ValueError("AnomalyAutoencoder model does not support global feature importances")

print('#'*50)
print('Time Scaling Experiments')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Interpretation Model: {interpretation}')
print(f'Sample Size: {dataset.X.shape[0]}')
print(f'Number of Features: {dataset.X.shape[1]}')
print('#'*50)

os.chdir('../')
cwd=os.getcwd()

# Save the results
path = cwd +"/experiments/results/"+dataset.name
if not os.path.exists(path):
    os.makedirs(path)
path_experiments = cwd +"/experiments/results/"+dataset.name+"/experiments"
if not os.path.exists(path_experiments):
    os.makedirs(path_experiments)

path_experiment = path_experiments + "/time_scaling"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)

path_experiment_model = path_experiment + "/" + model
if not os.path.exists(path_experiment_model):
    os.makedirs(path_experiment_model)
    
path_experiment_model_interpretation = path_experiment_model + "/" + interpretation
if not os.path.exists(path_experiment_model_interpretation):
    os.makedirs(path_experiment_model_interpretation)

path_experiment_model_fit_predict = path_experiment_model + "/" + 'fit_predict'
if not os.path.exists(path_experiment_model_fit_predict):
    os.makedirs(path_experiment_model_fit_predict)

# Fit Predict Experiment 
if fit_predict:
    fit_time,predict_time=fit_predict_experiment(I=I,dataset=dataset,n_runs=n_runs,model=I.name)

    print(f'Mean Fit Time: {fit_time}')
    print(f'Mean Predict Time: {predict_time}')

    time_dict={"fit_time":fit_time,"predict_time":predict_time}
    save_element(time_dict, path_experiment_model_fit_predict, filetype="pickle")

# Importances Experiment

if GFI:
    if interpretation == "NA":
        raise ValueError("Interpretation algorithm not specified")
    
    print('#'*50)
    print('Global Feature Importances Experiment')
    print('#'*50)

    _,imp_times = experiment_global_importances(I, dataset, n_runs=n_runs, p=contamination, interpretation=interpretation)

    print(f'Mean Importances Time: {imp_times}')
    imp_dict={"importances_time":imp_times}
    save_element(imp_dict, path_experiment_model_interpretation, filetype="pickle")





