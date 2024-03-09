import sys
import os
os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
#os.chdir('/Users/alessio/Documents/ExIFFI/experiments')
sys.path.append("..")
from collections import namedtuple

from utils_reboot.experiments import *
from utils_reboot.datasets import *
from utils_reboot.plots import *
from utils_reboot.utils import *


from model_reboot.EIF_reboot import ExtendedIsolationForest
import argparse
import pickle
import matplotlib.pyplot as plt

# Create the argument parser
parser = argparse.ArgumentParser(description='Test Global Importances')

# Add the arguments
parser.add_argument('--dataset_name', type=str, default='annthyroid', help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='../data/real/', help='Path to the dataset')
parser.add_argument("--change_ylim",action="store_true",help="If set, change the ylim of the plot")

# Parse the arguments
args = parser.parse_args()

dataset_name = args.dataset_name
dataset_path = args.dataset_path
change_ylim = args.change_ylim

dataset = Dataset(name=dataset_name, path=dataset_path)

path_contamination = os.getcwd()+'/results/'+ dataset.name +'/experiments/contamination/'
path_plots = os.getcwd()+'/results/'+ dataset.name +'/plots_new/contamination_plots'
colors = ["tab:green","tab:blue","tab:orange","tab:red","tab:purple"]
plt.style.use('default')
plt.rcParams['axes.facecolor'] = '#F2F2F2'
plt.grid(alpha = 0.7)
for i,model in enumerate(["DIF","EIF","EIF+","IF","AnomalyAutoencoder"]):
    path_contamination_model = path_contamination + model 
    file = get_most_recent_file(path_contamination_model)
    with open(file, "rb") as file:
        df = pickle.load(file)
    plt.plot(df[1],df[0].mean(axis=1),label=model,color=colors[i],marker='o')
    plt.fill_between(df[1], [np.percentile(x, 10) for x in df[0]], [np.percentile(x, 90) for x in df[0]],alpha=0.1, color=colors[i])

if change_ylim:
    plt.ylim(0,1.1)
else:
    plt.ylim(0,1)

plt.xlabel("Contamination",fontsize = 20)
plt.ylabel("Average Precision",fontsize = 20)
plt.legend()

t = time.localtime()
current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
nameplot = current_time+"_"+dataset.name+"_contamination_full_plot.pdf"

plt.savefig(path_plots+"/"+nameplot)
    

