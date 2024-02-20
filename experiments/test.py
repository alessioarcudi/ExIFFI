

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
# initialize contamination paths
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

