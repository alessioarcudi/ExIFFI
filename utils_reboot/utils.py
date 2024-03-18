import time
from typing import Type
import pickle
import numpy as np
import pandas as pd
import os
from collections import namedtuple
from model_reboot.EIF_reboot import ExtendedIsolationForest
from sklearn.ensemble import IsolationForest 
from pyod.models.dif import DIF as oldDIF
from pyod.models.auto_encoder import AutoEncoder as oldAutoEncoder

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score

Precisions = namedtuple("Precisions",["direct","inverse","dataset","model","value"])
NewPrecisions = namedtuple("NewPrecisions", ["direct", "inverse", "dataset", "model", "value", "aucfs"])
Precisions_random = namedtuple("Precisions_random",["random","dataset","model"])


class sklearn_IsolationForest(IsolationForest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "sklearn_IF"
    
    def predict(self, X):
        score=self.decision_function(X)
        return -1*score+0.5

class DIF(oldDIF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DIF"
    
    def predict(self, X):
        score=self.decision_function(X)
        return score
    
class AutoEncoder(oldAutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AnomalyAutoencoder"
    
    def predict(self, X):
        score=self.decision_function(X)
        return score
    

def save_element(element, directory_path, filename="", filetype="pickle"):
    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    filename = current_time + '_' + filename
    path = directory_path + '/' + filename
    if filetype == "pickle":
        with open(path+".pickle", 'wb') as fl:
            pickle.dump(element, fl)
    elif filetype == "npz":
        np.savez(path, element=element)
        
def get_most_recent_file(directory_path):
    # files = sorted(os.listdir(directory_path))
    # return directory_path+"/"+files[-1]
    files = sorted(os.listdir(directory_path), key=lambda x: os.path.getmtime(os.path.join(directory_path, x)), reverse=True)
    return os.path.join(directory_path, files[0])

def open_element(file_path, filetype="pickle"):
    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    if filetype == "pickle":
        with open(file_path, 'rb') as fl:
            element = pickle.load(fl)
    elif filetype == "npz":
        element = np.load(file_path)['element']
    return element

def fix_fs_file(dataset,model,interpretation,scenario):
    path=os.path.join(os.getcwd(),dataset.name,'experiments','feature_selection',model,interpretation,f'scenario_{str(scenario)}')
    file_path=get_most_recent_file(path)
    precs=open_element(file_path)[0]
    aucfs=sum(precs.inverse.mean(axis=1)-precs.direct.mean(axis=1))
    new_precs = NewPrecisions(direct=precs.direct,
                            inverse=precs.inverse,
                            dataset=precs.dataset,
                            model=precs.model,
                            value=precs.value,
                            aucfs=aucfs)
    save_element(new_precs, path, filetype="pickle")

def save_fs_prec(precs,path):
    aucfs=sum(precs.inverse.mean(axis=1)-precs.direct.mean(axis=1))
    new_precs = NewPrecisions(direct=precs.direct,
                            inverse=precs.inverse,
                            dataset=precs.dataset,
                            model=precs.model,
                            value=precs.value,
                            aucfs=aucfs)
    save_element(new_precs, path, filetype="pickle")

def save_fs_prec_random(precs,path):
    new_precs = Precisions_random(random=precs.random,
                            dataset=precs.dataset,
                            model=precs.model)
    save_element(new_precs, path, filetype="pickle")

def get_fs_file(dataset,model,interpretation,scenario):
    path=os.path.join(os.getcwd(),dataset.name,'experiments','feature_selection',model,interpretation,f'scenario_{str(scenario)}')
    file_path=get_most_recent_file(path)
    precs=open_element(file_path)
    return precs