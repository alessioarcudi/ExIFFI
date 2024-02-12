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

sys.path.append('../src')
from src.utils import load_preprocess

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# def performance(X,y,model=IsolationForest()):
#     """
#     Compute the classical Classification metrics for the IF and EIF model
#     --------------------------------------------------------------------------------
    
#     Parameters
#     ----------
#     X: np.array
#         Input dataset
#     y: np.array
#             Labels of the input dataset
#     model: Object
#         Anomaly Detection model, by default we put the vanilla version of IsolationForest (all the default parameters)
#     Returns
#     ----------
#     d: dict
#     Dictionary containing the Classification Performance of the Isolation Forest model
        
#     """
#     y_pred=model.predict(X)
#     d={}
#     d['Precision']=sklearn.metrics.precision_score(y,y_pred) 
#     d['Recall']=sklearn.metrics.recall_score(y,y_pred) 
#     d['f1 score']=sklearn.metrics.f1_score(y,y_pred) 
#     d['Accuracy']=sklearn.metrics.accuracy_score(y,y_pred) 
#     d['Balanced Accuracy']=sklearn.metrics.balanced_accuracy_score(y,y_pred) 
#     d['Average Precision']=sklearn.metrics.average_precision_score(y,y_pred) 
#     d['ROC AUC Score']=sklearn.metrics.roc_auc_score(y,y_pred) 
      
#     return d,y_pred

def performance(X,y,model=IsolationForest()):
    """
    Compute the classical Classification metrics for the IF and EIF model
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    X: np.array
        Input dataset
    y: np.array
            Labels of the input dataset
    model: Object
        Anomaly Detection model, by default we put the vanilla version of IsolationForest (all the default parameters)
    Returns
    ----------
    d: dict
    Dictionary containing the Classification Performance of the Isolation Forest model
        
    """
    y_pred=model.predict(X).astype(int)
    d=[]
    d.append(sklearn.metrics.precision_score(y,y_pred))
    d.append(sklearn.metrics.recall_score(y,y_pred))
    d.append(sklearn.metrics.f1_score(y,y_pred))
    d.append(sklearn.metrics.accuracy_score(y,y_pred)) 
    d.append(sklearn.metrics.balanced_accuracy_score(y,y_pred)) 
    d.append(sklearn.metrics.average_precision_score(y,y_pred))
    d.append(sklearn.metrics.roc_auc_score(y,y_pred))
      
    return d,y_pred

def collect_performance_df(dataset_names,
                        dataset_paths,
                        scaler,
                        model,
                        metric_names=['Precision', 'Recall', 'f1 score', 'Accuracy', 'Balanced Accuracy', 'Average Precision', 'ROC AUC Score']):
    
    dataset_perf=[]
    for name,path in zip(dataset_names,dataset_paths):
        X_train,X_test,X,y=load_preprocess(scaler,name,path)
        model.fit(X_train)
        dataset_perf.append(performance(X,y,model)[0])

    df_dataset_perf=pd.DataFrame(np.array(dataset_perf),columns=metric_names)
    df_dataset_perf.insert(0,'Dataset',dataset_names)

    return df_dataset_perf

def get_performance_dict(name,
                         X_train,
                         X_test,
                         y,
                         n_runs=10,
                         metric_dict={},
                         model=IsolationForest(),
                         model_name='IF',
                         metric_names=['Precision', 'Recall', 'f1 score', 'Accuracy', 'Balanced Accuracy', 'Average Precision', 'ROC AUC Score']):
    """
    Exploit the performance function to compute the Performance Report for an Anomaly Detection model on a specific dataset. 
    To avoid obtaining strange results because of the randomness of the executions, compute the performances n_runs times and save in the
    final dictionary the average value of all the metrics. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    name: str
            Dataset name
    X_train: pd.DataFrame
            Training Set
    X_test: pd.DataFrame
            Test Set
    y: np.array
            Predictions obtained with the Isolation Forest model using the if_predict function
    n_runs: int
            Number of executios, by default 10 
    model: Object
             Anomaly Detection model, by default we put the vanilla version of IsolationForest (all the default parameters)
    model_name: str
             Name of the model to insert in the resulting pd.DataFrame, by default IF for IsolationForest()
    Returns
    ----------
    d_perd: dict
             Dictionary containing the Classification Performance for the input model on the specified dataset 
    """

    metric_mat=np.zeros((n_runs,len(metric_names)))

    for i in range(n_runs): 
        model.fit(X_train)
        metric_mat[i,:]=np.array(performance(X_test,y,model)[0])

#     d_pred['name']=name  
#     d_pred['model']=model_name

    return metric_mat
    
def collect_performance(metrics_dict,
                        name,
                        X_train,
                        X_test,
                        y,
                        ):
    """
    Exploit the performance function compute the Performance Report for an Anomaly Detection model
    on a specific dataset for n_runs executions. 
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

