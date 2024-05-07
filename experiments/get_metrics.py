import os
import sys
sys.path.append('../')
cwd=os.getcwd()
import pickle
from utils_reboot.utils import *
from utils_reboot.datasets import *
import argparse
#from pyod.models.dif import DIF
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Performance Metrics')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--n_estimators', type=int, default=200, help='EIF parameter: n_estimators')
parser.add_argument('--max_depth', type=str, default='auto', help='EIF parameter: max_depth')
parser.add_argument('--max_samples', type=str, default='auto', help='EIF parameter: max_samples')
parser.add_argument('--contamination', type=npt.NDArray, default=np.linspace(0.0,0.1,10), help='Global feature importances parameter: contamination')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument('--pre_process',action='store_true', help='If set, preprocess the dataset')

def get_precision_file(dataset,model,scenario):    
    path=os.path.join(cwd+"/results/",dataset.name,'experiments','metrics',model,f'scenario_{str(scenario)}')
    file_path=get_most_recent_file(path)
    results=open_element(file_path)
    return results

# Parse the arguments
args = parser.parse_args()

# Access the arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
n_estimators = args.n_estimators
max_depth = args.max_depth
max_samples = args.max_samples
contamination = args.contamination
model = args.model
scenario = args.scenario
pre_process = args.pre_process

dataset = Dataset(dataset_name, path = dataset_path,feature_names_filepath='../data/')
dataset.drop_duplicates()

print('#'*50)
print('Performance Metrics Experiment')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Model: {model}')
print(f'Scenario: {scenario}')
print('#'*50)

# Downsample datasets with more than 7500 samples (i.e. diabetes shuttle and moodify)
if dataset.shape[0]>7500:
    dataset.downsample(max_samples=7500)

if scenario==2:
    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

# Preprocess the dataset
if pre_process:
    print("#"*50)
    print("\n\nPreprocessing the dataset...")
    print("#"*50)
    dataset.pre_process()
else:
    print("#"*50)
    print("\n\nDataset not preprocessed")
    dataset.initialize_train_test()
    print("#"*50)

#import ipdb; ipdb.set_trace()

if model == "EIF+":
    I = ExtendedIsolationForest(1, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "EIF":
    I = ExtendedIsolationForest(0, n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif (model == "IF"):
    I = IsolationForest(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
elif model == "DIF":
    I=DIF(max_samples=max_samples)
elif model == "AnomalyAutoencoder":
    I = AutoEncoder(hidden_neurons=[dataset.X.shape[1], 32, 32, dataset.X.shape[1]], contamination=0.1, epochs=50, random_state=42,verbose=0)

I.fit(dataset.X_train)

y_pred=I._predict(dataset.X_test,p=dataset.perc_outliers)
score=I.predict(dataset.X_test)

# y_pred=I.predict(dataset.X_test,contamination=dataset.perc_outliers)
# score=I.decision_function(dataset.X_test)


#print(f'\n\nPredictions of {model} for {dataset.name} in scenario {str(scenario)} with pre_process = {pre_process}:\n\n{y_pred}\n\n')
print(f'\n\nPrecision:{precision_score(dataset.y_test,y_pred)}')
print(f'\n\nRecall:{recall_score(dataset.y_test,y_pred)}')
print(f'\n\nAverage Precision:{average_precision_score(dataset.y_test,score)}')
print(f'\n\nROC AUC Score: {roc_auc_score(dataset.y_test,y_pred)}\n\n')
print("#"*50)


# print(f'Anomaly Score of the anomalous points: {score[dataset.y_test==1]}\n\n')
# print(f'Anomaly Score of the normal points: {score[dataset.y_test==0]}\n\n')
# print("#"*50)
# print(f'Mean Anomaly Score of the anomalous points: {np.mean(score[dataset.y_test==1])}\n\n')
# print(f'Mean Anomaly Score of the normal points: {np.mean(score[dataset.y_test==0])}\n\n')
# print("#"*50)

# print("#"*50)
# print(f'Performance values for {dataset.name} {model} scenario {str(scenario)}')
# print(get_precision_file(dataset,model,scenario).T)
# print("#"*50)