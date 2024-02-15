from typing import Type

import sys
sys.path.append('..')

import os
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from model_reboot.EIF_reboot import ExtendedIsolationForest
from utils_reboot.datasets import Dataset
import time
import pickle



def compute_global_importances(I: Type[ExtendedIsolationForest],
                               dataset: Type[Dataset],
                               n_runs:int = 10, 
                               p = 0.1,
                               #depth_based: bool = False,
                               pwd_imp_score: str = os.getcwd(),
                               pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
    """
    Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
    functions. 
    
    Parameters
    ----------
    X: Input Dataset,np.array of shape (n_samples,n_features)
    n_runs: Number of runs to perform in order to compute the Global Feature Importance Scores.  
    depth_based: Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance in Global_importance
    pwd_imp_score: Directory where the Importance Scores results will be saved as pkl files, by default the current working directory
    pwd_plt_data: Directory where the plot data results will be saved as pkl files, by default the current working directory        
                            
    Returns
    ----------
    imps: array of shape (n_samples,n_features) containing the local Feature Importance values for the samples of the input dataset X. 
    The array is also locally saved in a pkl file for the sake of reproducibility.
    plt_data: Dictionary containing the average Importance Scores values, the feature order and the standard deviations on the Importance Scores. 
    The dictionary is also locally saved in a pkl file for the sake of reproducibility.
    path_fi: Path of the pkl file containing the Importance Scores
    path_plt_data: Path of the pkl file containing the plt data    
    """

    fi=np.zeros(shape=(n_runs,dataset.X.shape[1]))
    for i in tqdm(range(n_runs)):
        
        I.fit(dataset.X)
        fi[i,:]=I.global_importances(dataset.X,p)
    """
    NON SO SAI
    # Handle the case in which there are some np.nan or np.inf values in the fi array
    if np.isnan(fi).any() or np.isinf(fi).any():
            #Substitute the np.nan values with 0 and the np.inf values with the maximum value of the fi array plus 1. 
            fi=np.nan_to_num(fi,nan=0,posinf=np.nanmax(fi[np.isfinite(fi)])+1)
    """
    
    # Save the Importance Scores in a npz file (more efficient than pkl files if we are using Python objects)
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    filename = current_time + '_GFI_p=' + str(p) + "_" + str(n_runs) + "runs_" +  dataset.name + '.npz'
    path_fi = pwd_imp_score  + '/' + filename
    np.savez(path_fi,fi=fi)
            
    """
    fi[fi==np.inf]=np.nan
    mean_imp=np.nanmean(fi,axis=0)
    std_imp=np.nanstd(fi,axis=0)
    """
    mean_imp = np.mean(fi,axis=0)
    std_imp = np.std(fi,axis=0)
    feat_ordered = mean_imp.argsort()
    mean_ordered = mean_imp[feat_ordered]
    std_ordered = std_imp[feat_ordered]

    plt_data={'Importances': mean_ordered,
                'feat_order': feat_ordered,
                'std': std_ordered}
    
    # Save the plt_data dictionary in a pkl file
    filename_plt = current_time + '_GFI_mean_' + str(n_runs) + "runs_" + dataset.name + '.pkl'
    path_plt_data = pwd_plt_data + '/' + filename_plt
    with open(path_plt_data, 'wb') as fl:
        pickle.dump(plt_data,fl)
    

    return fi,plt_data,path_fi,path_plt_data
    

def feature_selection(I: Type[ExtendedIsolationForest],
                      dataset: Type[Dataset],
                      importances_indexes: npt.NDArray,
                      n_runs: int = 10, 
                      p = 0.1,
                      ) -> tuple[np.array,dict,str,str]:
        for number_of_features_dropped in range(len(importances_indexes)):
            

        
        
        
        
        
        
        if feature_ranking_algorithm=="casual":
        precisionsEIF=[]
        l_o=list(range(dim))
        for i in tqdm(range(0,dim)):
            
            precisionEIF=[]
            for s in range(0,precision): #precision is equal to 10 in compare_feature_selection.py so here it is used 
                #just to do 10 instances of each model 
                EIF=None
                while not EIF:
                    l=np.random.choice(l_o,dim-i,replace=False)
                    X_red_train,y_red_train = X_train[:,l],y_train
                    X_red,y_red = X[:,l],y
                    X_red_train,y_red_train=drop_duplicates(X_red_train,y_red_train)
                    X_red,y_red = drop_duplicates(X_red,y_red)
                    EIF=ExtendedIF(n_trees=n_trees,plus=1)
                    EIF.fit(X_red_train)
                    score = EIF.predict(X_red)
                    avg_prec = sklearn.metrics.average_precision_score(y_red,score)
                precisionEIF.append(avg_prec)
            precisionsEIF.append(precisionEIF)

        #At the end save the result in a pkl file 
        #path = '../results/feature_selection/results/AvgPrecisions/'+ feature_ranking_algorithm + '_' + name
        #New put for Davide exeuctions
        path = 'c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\results\\feature_selection\\results\\Precisions_davide\\'+ feature_ranking_algorithm + '_' + name
        with open(path, 'wb') as f:
            pickle.dump([precisionsEIF], f)  

    #ExIFFI Feature Selection -> Remove every time the least important feature according to the ExIFFI feature importance score
    else:
        precisionsEIF=[]
        #Importances = importances[feature_ranking_algorithm].sort_values(by=[0],ascending=False)
        Importances = importances[feature_ranking_algorithm]
        for i in tqdm(range(0,dim)):
            precisionEIF=[]
            X_red_train,y_red_train=X_train[:,list(Importances.index)[:dim-i]],y_train
            X_red,y_red=X[:,list(Importances.index)[:dim-i]],y
            X_red_train,y_red_train=drop_duplicates(X_red_train,y_red_train)
            X_red,y_red = drop_duplicates(X_red,y_red)
            for s in range(precision):
            
                try:
                    EIF=ExtendedIF(n_trees=n_trees,plus=1)
                    EIF.fit(X_red_train)
                    score = EIF.predict(X_red)
                    avg_prec = sklearn.metrics.average_precision_score(y_red,score)
                except: 
                    avg_prec = np.nan
                precisionEIF.append(avg_prec)
            precisionsEIF.append(precisionEIF)

        #At the end save the result in a pkl file 
        #path = '../results/feature_selection/results/AvgPrecisions/'+ feature_ranking_algorithm + '_' + name
        path = 'c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\results\\feature_selection\\results\\Precisions_davide\\'+ feature_ranking_algorithm + '_' + name
        with open(path, 'wb') as f:
            pickle.dump([precisionsEIF], f)
    
