import sys
import os
cwd = os.getcwd()
sys.path.append("..")
from utils_reboot.datasets import *
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Data Loading')
parser.add_argument('--dataset_name', type=str, default='wine', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
args = parser.parse_args()
dataset_name = args.dataset_name
dataset_path = args.dataset_path

dataset = Dataset(dataset_name, path = dataset_path,feature_names_filepath='../data/')
dataset.drop_duplicates()

print('#'*50)
print('Dataset loaded')
print('#'*50)
print(f'Dataset: {dataset.name}')
print(f'Shape: {dataset.shape}')
print(f'Feature names: {dataset.feature_names}')
print(f'Contaminatino factor: {dataset.perc_outliers}')
print('#'*50)

