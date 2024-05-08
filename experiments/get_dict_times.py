import os
import sys
sys.path.append('../')
import pickle
import argparse
from utils_reboot.utils import *
from utils_reboot.datasets import *

parser = argparse.ArgumentParser(description='Test Performance Metrics')


parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')

os.chdir('../utils_reboot')
with open(os.getcwd() + "/time_scaling_test_if_exiffi.pickle", "rb") as file:
    dict_time = pickle.load(file)

def compute_mean_times(d,type,model,dataset_name):
    times=d[type][model][dataset_name]
    times=np.sort(times)
    times=times[int(len(times)*0.1):int(len(times)*0.9)]
    return np.mean(times)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path

dataset = Dataset(dataset_name, path = dataset_path)

# time_df=pd.DataFrame({

#     'model':['EIF+','EIF','IF','DIF','AE'],

#     'fit':[compute_mean_times(dict_time,'fit','EIF+',dataset.name),
#     compute_mean_times(dict_time,'fit','EIF',dataset.name),
#     compute_mean_times(dict_time,'fit','sklearn_IF',dataset.name),
#     compute_mean_times(dict_time,'fit','DIF',dataset.name),
#     compute_mean_times(dict_time,'fit','AnomalyAutoencoder',dataset.name)],

#     'predict':[compute_mean_times(dict_time,'predict','EIF+',dataset.name),
#     compute_mean_times(dict_time,'predict','EIF',dataset.name),
#     compute_mean_times(dict_time,'predict','sklearn_IF',dataset.name),
#     compute_mean_times(dict_time,'predict','DIF',dataset.name),
#     compute_mean_times(dict_time,'predict','AnomalyAutoencoder',dataset.name)]
# })

# print("#"*50)
# print(f'Execution time for fit and predict for {dataset.name}')
# print(time_df)
# print("#"*50)

# imp_df=pd.DataFrame({

#     'model':['DIFFI','EXIFFI','IF_EXIFFI','EXIFFI+','IF_RF','EIF_RF','EIF+_RF'],

#     'importances':[compute_mean_times(dict_time,'importances','DIFFI',dataset.name),
#                    compute_mean_times(dict_time,'importances','EXIFFI',dataset.name),
#                    compute_mean_times(dict_time,'importances','EXIFFI',dataset.name),
#                    compute_mean_times(dict_time,'importances','EXIFFI+',dataset.name),
#                    compute_mean_times(dict_time,'importances','RandomForest',dataset.name),
#                    compute_mean_times(dict_time,'importances','RandomForest',dataset.name),
#                    compute_mean_times(dict_time,'importances','RandomForest',dataset.name)]
# })

print("#"*50)
print(f'Execution time for importances computation for {dataset.name}')
#print(imp_df)
print(compute_mean_times(dict_time,'importances','IF_EXIFFI',dataset.name))
print("#"*50)