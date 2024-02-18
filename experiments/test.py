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

dataset = Dataset("wine", path = "../data/real/")
dataset.drop_duplicates()

I=ExtendedIsolationForest(True, n_estimators=100)

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

#----------------- GLOBAL IMPORTANCES -----------------#
# initialize global_importances paths
path_experiment = path_experiments + "/global_importances"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
path_experiment_matrices = path_experiment + "/matrices"
if not os.path.exists(path_experiment_matrices):
    os.makedirs(path_experiment_matrices)
path_experiment_plots = path_experiment + "/data_for_plots"
if not os.path.exists(path_experiment_plots):
    os.makedirs(path_experiment_plots)
    
#Compute global importances
full_importances, data_for_plots = compute_global_importances(I, dataset, 10, p=0.1)    
save_element(full_importances, path_experiment_matrices, filetype="npz")
save_element(data_for_plots, path_experiment_plots, filetype="pickle")

# plot global importances
most_recent_file = get_most_recent_file(path_experiment_matrices)
bar_plot(dataset, most_recent_file, path_plots, filetype="npz", show_plot=False)
most_recent_file = get_most_recent_file(path_experiment_plots)
score_plot(dataset, most_recent_file, path_plots, show_plot=False)



#----------------- LOCAL SCOREMAP -----------------#
# initialize local_scoremap paths
path_experiment = path_experiments + "/local_scoremap"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)

# Compute local scoremap



#----------------- FEATURE SELECTION -----------------#
# initialize feature_selection paths
path_experiment = path_experiments + "/feature_selection"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
path_experiment_matrices = path_experiment + "/matrices"
if not os.path.exists(path_experiment_matrices):
    os.makedirs(path_experiment_matrices)

# feature selection
most_recent_file = get_most_recent_file(path_experiment_plots)
data_for_plots = open_element(most_recent_file)
Precisions = namedtuple("Precisions",["direct","inverse","dataset","model"])
direct = feature_selection(I, dataset, data_for_plots["feat_order"], 10, inverse=False, random=False)
inverse = feature_selection(I, dataset, data_for_plots["feat_order"], 10, inverse=True, random=False)
data = Precisions(direct, inverse, dataset.name, I.name)
save_element(data, path_experiment_matrices, filetype="pickle")

#plot feature selection
most_recent_file = get_most_recent_file(path_experiment_matrices)
plot_feature_selection(most_recent_file, path_plots, show_plot=False)


#----------------- EVALUATE PRECISIONS OVER CONTAMINATION -----------------#
# initialize contamiination paths
path_experiment = path_experiments + "/contamination"
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
path_experiment_matrices = path_experiment + "/precisions"

# contamination evaluation
precisions = contamination_in_training_precision_evaluation(I, dataset, 10, 0.9)
save_element(precisions, path_experiment_matrices, filetype="pickle")

#plot contamination evaluation
precisions = open_element(get_most_recent_file(path_experiment_matrices))
plot_precision_over_contamination(precisions, path_plots, I.name, show_plot=False)

