import ast
import os
import sys

cwd = os.getcwd()
# os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
# os.chdir('/Users/alessio/Documents/ExIFFI/experiments')
sys.path.append("..")
import argparse
from collections import namedtuple

from model_reboot.EIF_reboot import ExtendedIsolationForest, IsolationForest
from utils_reboot.datasets import *
from utils_reboot.experiments import *
from utils_reboot.plots import *
from utils_reboot.utils import *

# Create the argument parser
parser = argparse.ArgumentParser(description="Test Local Importances")

# Add the arguments
parser.add_argument(
    "--dataset_name", type=str, default="wine", help="Name of the dataset"
)
parser.add_argument(
    "--dataset_path", type=str, default="../data/real/", help="Path to the dataset"
)
parser.add_argument("--plus", type=bool, default=True, help="EIF parameter: plus")
parser.add_argument(
    "--n_estimators", type=int, default=100, help="EIF parameter: n_estimators"
)
parser.add_argument(
    "--max_depth", type=str, default="auto", help="EIF parameter: max_depth"
)
parser.add_argument(
    "--max_samples", type=str, default="auto", help="EIF parameter: max_samples"
)
parser.add_argument(
    "--contamination",
    type=float,
    default=0.1,
    help="Global feature importances parameter: contamination",
)
parser.add_argument(
    "--n_runs",
    type=int,
    default=10,
    help="Global feature importances parameter: n_runs",
)
parser.add_argument(
    "--model",
    type=str,
    default="EIF+",
    help="Name of the interpretable AD model. Accepted values are: [IF,EIF,EIF+]",
)
parser.add_argument(
    "--interpretation",
    type=str,
    default="EXIFFI+",
    help="Name of the interpretation model. Accepted values are: [EXIFFI+,EXIFFI,DIFFI,RandomForest]",
)
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument(
    "--pre_process", type=bool, default=False, help="If set, preprocess the dataset"
)
parser.add_argument(
    "--feature1",
    type=lambda x: int(x) if x.isdigit() else x,
    help="First feature of the pair to plot in the importance map",
)
parser.add_argument(
    "--feature2",
    type=lambda x: int(x) if x.isdigit() else x,
    help="Second feature of the pair to plot in the importance map",
)


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
scenario = args.scenario
pre_process = args.pre_process
feature1 = args.feature1
feature2 = args.feature2

dataset = Dataset(dataset_name, path=dataset_path, feature_names_filepath="../data/")
dataset.drop_duplicates()

# import ipdb; ipdb.set_trace()

feats_plot = get_feature_indexes(dataset, feature1, feature2)

# Downsample datasets with more than 7500 samples (i.e. diabetes shuttle and moodify)
if dataset.shape[0] > 7500:
    dataset.downsample(max_samples=7500)

if scenario == 2:
    # dataset.split_dataset(train_size=0.8,contamination=0)
    dataset.split_dataset(train_size=1 - dataset.perc_outliers, contamination=0)

# Preprocess the dataset
if pre_process:
    print("#" * 50)
    print("Preprocessing the dataset...")
    print("#" * 50)
    dataset.pre_process()
else:
    print("#" * 50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#" * 50)

assert model in ["IF", "EIF", "EIF+", "ECOD"], "Interpretable AD model not recognized"
assert interpretation in [
    "EXIFFI+",
    "EXIFFI",
    "DIFFI",
    "RandomForest",
    "ECOD",
], "Interpretation not recognized"

if interpretation == "DIFFI":
    assert model == "IF", "DIFFI can only be used with the IF model"

if args.interpretation == "ECOD":
    assert args.model == "ECOD", "ECOD can only be used with the ECOD model"

if interpretation == "EXIFFI":
    assert (
        model == "EIF" or model == "IF"
    ), "EXIFFI can be used with the EIF and IF models"

if interpretation == "EXIFFI+":
    assert model == "EIF+", "EXIFFI+ can only be used with the EIF+ model"

if model == "IF":
    if interpretation == "EXIFFI":
        I = IsolationForest(
            n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples
        )
    elif interpretation == "DIFFI" or interpretation == "RandomForest":
        I = sklearn_IsolationForest(n_estimators=n_estimators, max_samples=max_samples)
elif model == "EIF":
    I = ExtendedIsolationForest(
        0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples
    )
elif model == "EIF+":
    I = ExtendedIsolationForest(
        1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples
    )
elif model == "ECOD":
    I = ECOD(contamination=contamination)

print("#" * 50)
print("Local Importances Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {model}")
print(f"Interpretation Model: {interpretation}")
print(f"Scenario: {scenario}")
print(
    f"Features to plot: {dataset.feature_names[feats_plot[0]]}, {dataset.feature_names[feats_plot[1]]}"
)
print("#" * 50)

os.chdir("../")
cwd = os.getcwd()

path_plots = cwd + "/experiments/results/" + dataset.name + "/plots_new/local_scoremaps"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

# ----------------- LOCAL SCOREMAP -----------------#
# Compute local scoremap
I.fit(dataset.X_train)
# print(I.avg_number_of_nodes)
if interpretation == "DIFFI":
    importance_map(
        dataset,
        I,
        feats_plot=feats_plot,
        path_plot=path_plots,
        col_names=dataset.feature_names,
        interpretation=interpretation,
        scenario=scenario,
        isdiffi=True,
    )
else:
    importance_map(
        dataset,
        I,
        feats_plot=feats_plot,
        path_plot=path_plots,
        col_names=dataset.feature_names,
        interpretation=interpretation,
        scenario=scenario,
    )
