import numpy as np
import pandas as pd
import os 
import pickle
from tqdm import tqdm
import sys;
#sys.path.append("..//models")
from models.Extended_IF import *
from sklearn.metrics import average_precision_score
from sklearn.ensemble import IsolationForest
import matplotlib.patches as mpatches

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def evaluate_precisions(X_train,X_test,y,name):
    """
    Compute the Average Precision metric for the IF,EIF and EIF_plus models 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    X_train: pd.DataFrame
            Training Set
    X_test: pd.DataFrame
            Test Set
    y: np.array
            Dataset labels
    name: string
            Dataset name

    Returns
    ----------
    precision_IF,precision_EIF,precision_EIF_plus: float
            For each model the Average Precision score is returned
        
    """
    EIF=ExtendedIF(n_trees=300,plus=0)
    EIF.fit(X_train)
    EIF=ExtendedIF(n_trees=300,plus=1)
    EIF_plus.fit(X_train)
    iforest = IsolationForest(n_estimators=300,max_samples=256)
    iforest.fit(X_train)
    score = EIF.predict(X_test)
    precision_EIF = average_precision_score(y,score)
    score_plus = EIF_plus.predict(X_test)
    precision_EIF_plus = average_precision_score(y,score_plus)
    scoreif = -1*iforest.score_samples(X_test)+0.5
    precision_IF = average_precision_score(y,scoreif)
    return precision_IF,precision_EIF,precision_EIF_plus

def collect_precisions(Precisions_scores,name,X_train,X_test,y):
    """
    Exploiting the evaluate_precisions function the Average Precision scores for the IF,EIF and EIF_plus
    models are coputed 10 times and inserted in a dictionary. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    Precisions_scores: dict
            Dictionary containing the Precisions_scores of the IF,EIF and EIF_plus models for all the different datasets.
    name: string
            Dataset name 
    X_train: pd.DataFrame
            Training Set
    X_test: pd.DataFrame
            Test Set
    y: np.array
            Dataset labels
    Returns
    ----------
    Precisions_scores: dict
            Dictionary containing the Precisions_scores of the IF,EIF and EIF_plus models for all the different datasets.
        
    """
    Precisions_scores[name]={}
    Precisions_scores[name]['IF']=[]
    Precisions_scores[name]['EIF']=[]
    Precisions_scores[name]['EIF_plus']=[]
    for i in tqdm(range(10)):
        precision_IF,precision_EIF,precision_EIF_plus = evaluate_precisions(X_train,X_test,y,name)
        Precisions_scores[name]['IF'].append(precision_IF)
        Precisions_scores[name]['EIF'].append(precision_EIF)
        Precisions_scores[name]['EIF_plus'].append(precision_EIF_plus)
    return Precisions_scores


def adjacent_values(vals, q1, q3):
    """
    This is an auxiliary function used for the production of the Average Precision Violin Plots presented in the paper.  
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    vals: np.array
            Sequence of values. In particualr here the Average Precision scores obtained on the 10 executions performed 
            in the collect_precisions function are reported
    q1: float
            First quartile of the Average Precisions scores distribution
    q3: float
            Third quartile of the Average Precisions scores distribution
    Returns
    ----------
    lower_adjacent_value,upper_adjacent_value: float
            Lower and upper adjacent values for the Average Precision score distribution 
        
    """
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

