import os
import sys
sys.path.append('../')
cwd=os.getcwd()
import pickle
from utils_reboot.utils import *
from utils_reboot.datasets import *
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Performance Metrics')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument('--model', type=str, default="EIF", help='Model to use: IF, EIF, EIF+')
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")

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
model = args.model
scenario = args.scenario

dataset = Dataset(dataset_name, path = dataset_path)

print("#"*50)
print(f'Performance values for {dataset.name} {model} scenario {str(scenario)}')
print(get_precision_file(dataset,model,scenario).T)
print("#"*50)