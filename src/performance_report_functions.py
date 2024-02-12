import numpy as np
import pandas as pd
import os 
import pickle
from tqdm import tqdm
import sys;
sys.path.append("../models")
from models.Extended_IF import *
from models.forests import * 
import sklearn
from sklearn.metrics import precision_score,recall_score,accuracy_score,balanced_accuracy_score,f1_score,average_precision_score,roc_auc_score
from sklearn.ensemble import IsolationForest
import matplotlib.patches as mpatches

sys.path.append("../experiments")
from experiments.test_model import get_model 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def if_predict(score,p):
    """
    Obtain the predictions for the Isolation Forest model 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    score: np.array
            Anomaly Score values for the input samples
    p: float
            Contamination factor of the input dataset
    Returns
    ----------
    y: np.array
    Anomaly/Not Anomaly predictions are contained in y: 0 for inliers, 1 for outliers
        
    """
    y=score>np.sort(score)[::-1][int(p*len(score))]
    return y.astype(int)

def performance(model,X,y):
    """
    Compute the classical Classification metrics for a PyOD model
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    model: PyOD Model 
        Instance of the PyOD model
    X: np.array
        Input dataset
    y: np.array
        Dataset labels
    Returns
    ----------
    d: dict
    Dictionary containing the Classification Performance of the model

    """

    y_pred=model.predict(X)
    d={}
    d['Precision']=sklearn.metrics.precision_score(y,y_pred) 
    d['Recall']=sklearn.metrics.recall_score(y,y_pred) 
    d['f1 score']=sklearn.metrics.f1_score(y,y_pred) 
    d['Accuracy']=sklearn.metrics.accuracy_score(y,y_pred) 
    d['Balanced Accuracy']=sklearn.metrics.balanced_accuracy_score(y,y_pred) 
    d['Average Precision']=sklearn.metrics.average_precision_score(y,y_pred) 
    d['ROC AUC Score']=sklearn.metrics.roc_auc_score(y,y_pred) 
    return d,y_pred

    

def performance_if(y,score):
    """
    Compute the classical Classification metrics for the Isolation Forest model
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    y: np.array
            Predictions obtained with the Isolation Forest model using the if_predict function
    score: np.array
            Anomaly Score values for the input samples
    Returns
    ----------
    d: dict
    Dictionary containing the Classification Performance of the Isolation Forest model
        
    """
    p=sum(y)/len(y)
    y_pred=if_predict(score,p)
    d={}
    d['Precision']=sklearn.metrics.precision_score(y,y_pred) 
    d['Recall']=sklearn.metrics.recall_score(y,y_pred) 
    d['f1 score']=sklearn.metrics.f1_score(y,y_pred) 
    d['Accuracy']=sklearn.metrics.accuracy_score(y,y_pred) 
    d['Balanced Accuracy']=sklearn.metrics.balanced_accuracy_score(y,y_pred) 
    d['Average Precision']=sklearn.metrics.average_precision_score(y,y_pred) 
    d['ROC AUC Score']=sklearn.metrics.roc_auc_score(y,y_pred) 
    return d

def performance_eif(y,score,X_test,model):
    """
    Compute the classical Classification metrics for the EIF/EIF_plus model
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    y: np.array
            Predictions obtained with the Isolation Forest model using the if_predict function
    score: np.array
            Anomaly Score values for the input samples
    X_test: pd.DataFrame
            Test Set
    model: Object
            Instance of the model (EIF or EIF_plus)
    Returns
    ----------
    d: dict
    Dictionary containing the Classification Performance for the EIF/EIF_plus model
        
    """
    p=sum(y)/len(y)
    y_pred=model._predict(X_test,p).astype(int)
    d={}
    d['Precision']=sklearn.metrics.precision_score(y,y_pred) 
    d['Recall']=sklearn.metrics.recall_score(y,y_pred)
    d['f1 score']=sklearn.metrics.f1_score(y,y_pred)
    d['Accuracy']=sklearn.metrics.accuracy_score(y,y_pred)
    d['Balanced Accuracy']=sklearn.metrics.balanced_accuracy_score(y,y_pred)
    d['Average Precision']=sklearn.metrics.average_precision_score(y,score)
    d['ROC AUC Score']=sklearn.metrics.roc_auc_score(y,score)
    return d

def evaluate_performance(X_train,X_test,y):
    """
    Exploit the performance_if and performance_eif functions to compute the Performance Report for the 
    IF,EIF and EIF_plus models on a specific dataset
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    X_train: pd.DataFrame
            Training Set
    X_test: pd.DataFrame
            Test Set
    y: np.array
            Predictions obtained with the Isolation Forest model using the if_predict function
    Returns
    ----------
    metrics_if,metrics_eif,metrics_eif_plus: tuple of dict
    Dictionary containing the Classification Performance for the IF,EIF and EIF_plus model
        
    """
    EIF=ExtendedIsolationForest(n_estimators=300,plus=0)
    EIF.fit(X_train)

    EIF_plus=ExtendedIsolationForest(n_estimators=300,plus=1)
    EIF_plus.fit(X_train)

    IF=IsolationForest(n_estimators=300,max_samples=min(len(X_train),256))
    IF.fit(X_train)

    score_if=-1*IF.score_samples(X_test)+0.5
    score_eif=EIF.predict(X_test)
    score_eif_plus=EIF_plus.predict(X_test)

    metrics_if=performance_if(y,score_if)
    metrics_eif=performance_eif(y,score_eif,X_test,EIF)
    metrics_eif_plus=performance_eif(y,score_eif_plus,X_test,EIF_plus)

    return metrics_if,metrics_eif,metrics_eif_plus
    
def collect_performance(metrics_dict,name,X_train,X_test,y):
    """
    Exploit the evaluate_performance function compute the Performance Report for the 
    IF,EIF and EIF_plus models on a specific dataset for 10 executions. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    metrics_dict: dict
            Dictionary containing the Performance metrics for all the different datasets. 
    name: string
            Dataset name
    X_train: pd.DataFrame
            Training Set
    X_test: pd.DataFrame
            Test Set
    y: np.array
            Predictions obtained with the Isolation Forest model using the if_predict function
    Returns
    ----------
    metrics_dict: dict
    Dictionary containing the Performance metrics for all the different datasets. 
        
    """
    metrics_dict[name]={}
    metrics_dict[name]["IF"]={}
    metrics_dict[name]["EIF"]={}
    metrics_dict[name]["EIF_plus"]={}
    metric_names=['Precision', 'Recall', 'f1 score', 'Accuracy', 'Balanced Accuracy', 'Average Precision', 'ROC AUC Score']

    for metric_name in metric_names:
        metrics_dict[name]['IF'][metric_name]=[]
        metrics_dict[name]['EIF'][metric_name]=[]
        metrics_dict[name]['EIF_plus'][metric_name]=[]


    for i in tqdm(range(10)):
        metrics_if,metrics_eif,metrics_eif_plus=evaluate_performance(X_train,X_test,y)

        for metric_name in metric_names:
            metrics_dict[name]['IF'][metric_name].append(metrics_if[metric_name])
            metrics_dict[name]['EIF'][metric_name].append(metrics_eif[metric_name])
            metrics_dict[name]['EIF_plus'][metric_name].append(metrics_eif_plus[metric_name])

    for metric_name in metric_names:
        metrics_dict[name]['IF'][metric_name+'_avg']=np.mean(np.array(metrics_dict[name]['IF'][metric_name]))
        metrics_dict[name]['EIF'][metric_name+'_avg']=np.mean(np.array(metrics_dict[name]['EIF'][metric_name]))
        metrics_dict[name]['EIF_plus'][metric_name+'_avg']=np.mean(np.array(metrics_dict[name]['EIF_plus'][metric_name]))
     
    
    return metrics_dict

# Evaluate the model performances 

def evaluate_model(model_name, X_train, X_test, y, name, save_dir, filename=None):

    # Fit the model
    print(f"Fitting {name} model")
    model=get_model(model_name)
    model.fit(X_train)

    # Compute the performance metrics
    print(f"Computing performance metrics for {name} model")
    if model_name=='IF':
        score=model.predict(X_test)
        perf=performance_if(y,score)
    elif model_name=='EIF' or model_name=='EIF+':
        score=model.predict(X_test)
        perf=performance_eif(y,score,X_test,model)

    return perf