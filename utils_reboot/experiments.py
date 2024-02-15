from typing import Type

import sys
sys.path.append('..')

import os
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import copy

from model_reboot.EIF_reboot import ExtendedIsolationForest
from utils_reboot.datasets import Dataset
import sklearn



def compute_global_importances(I: Type[ExtendedIsolationForest],
                               dataset: Type[Dataset],
                               n_runs:int = 10, 
                               p = 0.1) -> tuple[np.array,dict,str,str]:

    fi=np.zeros(shape=(n_runs,dataset.X.shape[1]))
    for i in tqdm(range(n_runs)):
        I.fit(dataset.X)
        fi[i,:]=I.global_importances(dataset.X,p)
            
    mean_imp = np.mean(fi,axis=0)
    std_imp = np.std(fi,axis=0)
    feat_ordered = mean_imp.argsort()
    mean_ordered = mean_imp[feat_ordered]
    std_ordered = std_imp[feat_ordered]

    plt_data={'Importances': mean_ordered,
                'feat_order': feat_ordered,
                'std': std_ordered}
    return fi,plt_data
    

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


