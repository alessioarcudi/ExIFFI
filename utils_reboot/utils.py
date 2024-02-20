

import time
import pickle
import numpy as np
import pandas as pd
import os
from collections import namedtuple

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score

Precisions = namedtuple("Precisions",["direct","inverse","dataset","model"])


def save_element(element, directory_path, filename="", filetype="pickle"):
    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    filename = current_time + '_' + filename
    path = directory_path + '/' + filename
    if filetype == "pickle":
        with open(path, 'wb') as fl:
            pickle.dump(element, fl)
    elif filetype == "npz":
        np.savez(path, element=element)
        
def get_most_recent_file(directory_path):
    files = sorted(os.listdir(directory_path))
    return directory_path+"/"+files[-1]

def open_element(file_path, filetype="pickle"):
    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    if filetype == "pickle":
        with open(file_path, 'rb') as fl:
            element = pickle.load(fl)
    elif filetype == "npz":
        element = np.load(file_path)['element']
    return element

def performance(y_pred:np.array,
                y_true:np.array,
                model_name:str,
                dataset_name:str,
                contamination:float=0.1,
                train_size:float=0.8,
                filename:str="",
                path:str="/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments"
                ) -> pd.DataFrame: 
    
    # In path insert the local put up to the experiments folder:
    # For Davide → /home/davidefrizzo/Desktop/PHD/ExIFFI/experiments
    # For Alessio → /Users/alessio/Documents/ExIFFI

    
    df=pd.DataFrame({
        "Model": model_name,
        "Dataset": dataset_name,
        "Contamination": contamination,
        "Train Size": train_size,
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "f1 score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Average Precision": average_precision_score(y_true, y_pred),
        "ROC AUC Score": roc_auc_score(y_true, y_pred)
    }, index=[pd.Timestamp.now()])

    path=path + f"/results/{dataset_name}/experiments/metrics/{model_name}/"

    if not os.path.exists(path):
        os.makedirs(path)
    
    save_element(df, path, filename)
    
    return df