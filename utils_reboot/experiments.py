from typing import Type

import sys
sys.path.append('..')

import os
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import copy

from model_reboot.EIF_reboot import ExtendedIsolationForest
from models.interpretability_module import *
from utils_reboot.datasets import Dataset
import sklearn
from sklearn.ensemble import IsolationForest

def compute_global_importances(I: Type[ExtendedIsolationForest],
                        dataset: Type[Dataset],
                        isdiffi:bool=False,
                        p = 0.1,
                        fit_model = True) -> np.array: 
    if fit_model:
        I.fit(dataset.X)             
    if isdiffi:
        fi,_=diffi_ib(I,dataset.X)
    else:
        fi=I.global_importances(dataset.X,p)
    return fi
                        
def experiment_global_importances(I: Type[ExtendedIsolationForest],
                               dataset: Type[Dataset],
                               isdiffi:bool=False,
                               n_runs:int = 10, 
                               p = 0.1) -> tuple[np.array,dict,str,str]:

    fi=np.zeros(shape=(n_runs,dataset.X.shape[1]))
    for i in tqdm(range(n_runs)):
        fi[i,:]=compute_global_importances(I,
                        dataset,
                        isdiffi=isdiffi,
                        p = p)
    return fi

def compute_plt_data(imp_path):

    fi = np.load(imp_path)['element']
    mean_imp = np.mean(fi,axis=0)
    std_imp = np.std(fi,axis=0)
    feat_ordered = mean_imp.argsort()
    mean_ordered = mean_imp[feat_ordered]
    std_ordered = std_imp[feat_ordered]

    plt_data={'Importances': mean_ordered,
                'feat_order': feat_ordered,
                'std': std_ordered}
    return plt_data
    

def feature_selection(I: Type[ExtendedIsolationForest],
                      dataset: Type[Dataset],
                      importances_indexes: npt.NDArray,
                      n_runs: int = 10, 
                      inverse: bool = True,
                      random: bool = False
                      ) -> tuple[np.array,dict,str,str]:
        dataset_shrinking = copy.deepcopy(dataset)
        d = dataset.X.shape[1]
        precisions = np.zeros(shape=(len(importances_indexes),n_runs))
        for number_of_features_dropped in tqdm(range(len(importances_indexes))):
            if random:
                importances_indexes = np.random.choice(importances_indexes, len(importances_indexes), replace=False)
            dataset_shrinking.X = dataset.X[:,importances_indexes[:d-number_of_features_dropped]] if not inverse else dataset.X[:,importances_indexes[number_of_features_dropped:]]
            dataset_shrinking.y = dataset.y
            dataset_shrinking.drop_duplicates()
            runs = np.zeros(n_runs)
            for run in range(n_runs):
                try:
                    I.fit(dataset_shrinking.X)
                    score = I.predict(dataset_shrinking.X)
                    avg_prec = sklearn.metrics.average_precision_score(dataset_shrinking.y,score)
                    runs[run] = avg_prec
                except:
                    runs[run] = np.nan
            precisions[number_of_features_dropped] = runs
        return precisions
    

def contamination_in_training_precision_evaluation(I: Type[ExtendedIsolationForest],
                                                   dataset: Type[Dataset],
                                                   n_runs: int = 10,
                                                   train_size = 0.8,
                                                   contamination_values: npt.NDArray = np.linspace(0.0,0.1,10),
                                                   compute_global_importances:bool=False,
                                                   isdiffi:bool=False,
                                                   ) -> tuple[np.array,dict,str,str]:
    precisions = np.zeros(shape=(len(contamination_values),n_runs))
    if compute_global_importances:
        importances = np.zeros(shape=(len(contamination_values),n_runs,dataset.X.shape[1]))
    for i,contamination in tqdm(enumerate(contamination_values)):
        for run in range(n_runs):
            dataset.split_dataset(train_size,contamination)
            try:
                I.fit(dataset.X_train)
                if compute_global_importances:
                    importances[i,run,:] = compute_global_importances(I,dataset,isdiffi=isdiffi,p=contamination,fit_model=False)
                score = I.predict(dataset.X)
                avg_prec = sklearn.metrics.average_precision_score(dataset.y,score)
                precisions[i,run] = avg_prec
            except:
                precisions[i,run] = np.nan
                if compute_global_importances:
                    importances[i,run,:] = np.array([np.nan]*dataset.X.shape[1])
    if compute_global_importances:
        return precisions,importances
    return precisions


