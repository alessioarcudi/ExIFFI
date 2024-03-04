from typing import Type

import sys
sys.path.append('..')

import os
os.chdir("../")
cwd = os.getcwd()
os.chdir("experiments")
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import copy

from model_reboot.EIF_reboot import ExtendedIsolationForest
from model_reboot.interpretability_module import *
from utils_reboot.datasets import Dataset
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor

import pickle
import time

filename = cwd + "/utils_reboot/new_time.pickle"

if not os.path.exists(filename):
    dict_time = {"fit":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}}, 
            "predict":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}},
            "importances":{"EXIFFI+":{},"EXIFFI":{},"DIFFI":{},"RandomForest":{}}}
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
               
with open(filename, "rb") as file:
    dict_time = pickle.load(file)

    

def compute_global_importances(I: Type[ExtendedIsolationForest],
                        dataset: Type[Dataset],
                        p = 0.1,
                        interpretation="EXIFFI",
                        model = "EIF+",
                        fit_model = True) -> np.array: 
    if fit_model:
        I.fit(dataset.X_train)             
    if interpretation=="DIFFI":
        fi,_=diffi_ib(I,dataset.X)
    elif interpretation=="EXIFFI" or interpretation=='EXIFFI+':
        fi=I.global_importances(dataset.X,p)
    elif interpretation=="RandomForest":
        rf = RandomForestRegressor()
        rf.fit(dataset.X, I.predict(dataset.X))
        fi = rf.feature_importances_
    return fi
                        
def experiment_global_importances(I: Type[ExtendedIsolationForest],
                               dataset: Type[Dataset],
                               n_runs:int = 10, 
                               p = 0.1,
                               model = "EIF+",
                               interpretation="EXIFFI") -> tuple[np.array,dict,str,str]:


    fi=np.zeros(shape=(n_runs,dataset.X.shape[1]))
    for i in tqdm(range(n_runs)):
        start_time = time.time()
        fi[i,:]=compute_global_importances(I,
                        dataset,
                        p = p,
                        interpretation=interpretation,
                        model = model)
        gfi_time = time.time() - start_time
        if i>3:
            dict_time["importances"][interpretation].setdefault(dataset.name, []).append(gfi_time)
            #print(f'Added time {str(gfi_time)} to time dict')
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
    return fi

def compute_plt_data(imp_path):

    try:
        fi = np.load(imp_path)['element']
    except:
        print("Error: importances file should be npz")
    # Handle the case in which there are some np.nan in the fi array
    if np.isnan(fi).any():
        #Substitute the np.nan values with 0  
        #fi=np.nan_to_num(fi,nan=0)
        mean_imp = np.nanmean(fi,axis=0)
        std_imp = np.nanstd(fi,axis=0)
    else:
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
                    if dataset.X.shape[1] == dataset_shrinking.X.shape[1]:
                        
                        start_time = time.time()
                        I.fit(dataset_shrinking.X)
                        fit_time = time.time() - start_time
                        
                        if run >3:
                            dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time)
                        start_time = time.time()
                        score = I.predict(dataset_shrinking.X)
                        predict_time = time.time() - start_time
                        
                        if run >3:                        
                            dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)
                    else:
                        I.fit(dataset_shrinking.X)
                        score = I.predict(dataset_shrinking.X)
                    avg_prec = sklearn.metrics.average_precision_score(dataset_shrinking.y,score)
                    runs[run] = avg_prec
                except:
                    runs[run] = np.nan
            precisions[number_of_features_dropped] = runs
        
        with open(filename, "wb") as file:
            pickle.dump(dict_time, file)
        return precisions
    

def contamination_in_training_precision_evaluation(I: Type[ExtendedIsolationForest],
                                                   dataset: Type[Dataset],
                                                   n_runs: int = 10,
                                                   train_size = 0.8,
                                                   contamination_values: npt.NDArray = np.linspace(0.0,0.1,10),
                                                   compute_GFI:bool=False,
                                                   interpretation:str="EXIFFI"
                                                   ) -> tuple[np.array,dict,str,str]:

    precisions = np.zeros(shape=(len(contamination_values),n_runs))
    if compute_GFI:
        importances = np.zeros(shape=(len(contamination_values),n_runs,len(contamination_values),dataset.X.shape[1]))
    for i,contamination in tqdm(enumerate(contamination_values)):
        for j in range(n_runs):
            dataset.split_dataset(train_size,contamination)
            
            start_time = time.time()
            I.fit(dataset.X_train)
            fit_time = time.time() - start_time
            
            if j>3:
                try:
                    dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time)
                except:
                    print('Model not recognized: creating a new key in the dict_time for the new model')
                    dict_time["fit"].setdefault(I.name, {}).setdefault(dataset.name, []).append(fit_time)
            
            if compute_GFI:
                for k,c in enumerate(contamination_values):
                    start_time = time.time()
                    importances[i,j,k,:] = compute_global_importances(I,
                                                                    dataset,
                                                                    p=c,
                                                                    interpretation=interpretation,
                                                                    fit_model=False)
                    gfi_time = time.time() - start_time
                    if k>3: 
                        dict_time["importances"][interpretation].setdefault(dataset.name, []).append(gfi_time)
                    
            start_time = time.time()
            score = I.predict(dataset.X)
            predict_time = time.time() - start_time
            if j>3:
                try:
                    dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)
                except:
                    print('Model not recognized: creating a new key in the dict_time for the new model')
                    dict_time["predict"].setdefault(I.name, {}).setdefault(dataset.name, []).append(predict_time)
            
            avg_prec = sklearn.metrics.average_precision_score(dataset.y,score)
            precisions[i,j] = avg_prec
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
    if compute_GFI:
        return precisions,importances
    return precisions


