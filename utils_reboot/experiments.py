from typing import Type,Union

import sys
sys.path.append('..')

import os
os.chdir("../")
cwd = os.getcwd()
os.chdir("experiments")
import numpy as np
import numpy.typing as npt
from tqdm import tqdm,trange
import copy

from model_reboot.EIF_reboot import ExtendedIsolationForest
from model_reboot.interpretability_module import *
from utils_reboot.datasets import Dataset
from utils_reboot.utils import save_element
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score
import shap

import pickle
import time
import pandas as pd

filename = cwd + "/utils_reboot/time_scaling_test_if_exiffi.pickle"
#filename = cwd + "/utils_reboot/time_scaling_test_dei.pickle"

# dict_time = {1:{"fit":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}}, 
#         "predict":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}},
#         "importances":{"EXIFFI+":{},"EXIFFI":{},"DIFFI":{},"RandomForest":{}}},
#         2:{"fit":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}}, 
#         "predict":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}},
#         "importances":{"EXIFFI+":{},"EXIFFI":{},"DIFFI":{},"RandomForest":{}}}}

if not os.path.exists(filename):

    dict_time = {"fit":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}}, 
            "predict":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}},
            "importances":{"EXIFFI+":{},"EXIFFI":{},"IF_EXIFFI":{},"DIFFI":{},"RandomForest":{}}}
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
               
with open(filename, "rb") as file:
    dict_time = pickle.load(file)

    

def compute_global_importances(I: Type[ExtendedIsolationForest],
                        dataset: Type[Dataset],
                        p = 0.1,
                        interpretation="EXIFFI+",
                        fit_model = True) -> np.array: 
    
    """
    Compute the global feature importances for an interpration model on a specific dataset.

    Args:
        I: The AD model.
        dataset: Input dataset.
        p: The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        interpretation: Name of the interpretation method to be used. Defaults to "EXIFFI+".
        fit_model: Whether to fit the model on the dataset. Defaults to True.

    Returns:
        The global feature importance vector.

    """

    if fit_model:
        I.fit(dataset.X_train)        
    if interpretation=="DIFFI":
        fi,_=diffi_ib(I,dataset.X_test)
    elif interpretation=="EXIFFI" or interpretation=='EXIFFI+':
        fi=I.global_importances(dataset.X_test,p)
    elif interpretation=="RandomForest":
        rf = RandomForestRegressor()
        rf.fit(dataset.X_test, I.predict(dataset.X_test))
        fi = rf.feature_importances_
    return fi

def fit_predict_experiment(I: Type[ExtendedIsolationForest],
                            dataset: Type[Dataset],
                            n_runs:int = 40,
                            model='EIF+') -> tuple[float,float]:
    
    """
    Fit and predict the model on the dataset for a number of runs and keep track of the fit and predict times.

    Args:
        I: The AD model.
        dataset: Input dataset.
        n_runs: The number of runs. Defaults to 40.
        model: The name of the model. Defaults to 'EIF+'.

    Returns:
        The average fit and predict time.
    """

    fit_times = []
    predict_times = []
    
    for i in trange(n_runs):
        start_time = time.time()
        I.fit(dataset.X_train)
        fit_time = time.time() - start_time
        if i>3:  
            fit_times.append(fit_time)
            dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time) 
        
        start_time = time.time()
        if model in ['EIF','EIF+']:
            _=I._predict(dataset.X_test,p=dataset.perc_outliers)
            predict_time = time.time() - start_time
        elif model in ['sklearn_IF','DIF','AnomalyAutoencoder']:
            _=I.predict(dataset.X_test)
            predict_time = time.time() - start_time

        if i>3:
            predict_times.append(predict_time)
            dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)

    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)

    return np.mean(fit_times), np.mean(predict_times)
                        
def experiment_global_importances(I:Type[ExtendedIsolationForest],
                               dataset:Type[Dataset],
                               n_runs:int=10, 
                               p:float=0.1,
                               model:str="EIF+",
                               interpretation:str="EXIFFI+"
                               ) -> tuple[np.array,dict,str,str]:
    
    """
    Compute the global feature importances for an interpration model on a specific dataset for a number of runs.

    Args:
        I: The AD model.
        dataset: Input dataset.
        n_runs: The number of runs. Defaults to 10.
        p: The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        model: The name of the model. Defaults to 'EIF+'.
        interpretation: Name of the interpretation method to be used. Defaults to "EXIFFI+".
    
    Returns:
        The global feature importances vectors for the different runs and the average importances times.
    """
    fi=np.zeros(shape=(n_runs,dataset.X.shape[1]))
    imp_times=[]
    for i in tqdm(range(n_runs)):
        start_time = time.time()
        fi[i,:]=compute_global_importances(I,
                        dataset,
                        p = p,
                        interpretation=interpretation)
        gfi_time = time.time() - start_time
        if i>3:
            imp_times.append(gfi_time)
            if (model=="IF") and (interpretation=="EXIFFI"):
                dict_time["importances"]["IF_EXIFFI"].setdefault(dataset.name, []).append(gfi_time)
            else:
                dict_time["importances"][interpretation].setdefault(dataset.name, []).append(gfi_time)
            #print(f'Added time {str(gfi_time)} to time dict')
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
    return fi,np.mean(imp_times)

# Score function for EIF/EIF+ ACME 
def EIF_score_function(model,data):
    return model.predict(data)

# Score function for IF ACME
def IF_score_function(model,data):
    return 0.5 * (- model.decision_function(data) + 1)

def compute_imp_time_kernelSHAP(I: Type[ExtendedIsolationForest],
                                 dataset: Type[Dataset],
                                 p: float = 0.1,
                                 background:float=0.1,
                                 pre_process:float=False,
                                 scenario:int=2) -> float: 
    
    """
    Compute the time to compute the local feature importances for an anomalous point using the KernelSHAP method.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Type[Dataset]): Input dataset.
        background (float): The percentage of the dataset to use as background. Defaults to 0.1.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        pre_process (bool): Whether to pre process the dataset after computing the downsampled version according to the background. Defaults to False.
        scenario (int): The scenario of the experiment. Defaults to 2.
    
    Returns:
        The time to compute the local feature importances for a single anomaly 
    """

    I.fit(dataset.X_train)

    y_pred=I._predict(dataset.X_test,p).astype(int)
    anomalies=dataset.X_test[np.where(y_pred==1)[0]]
    anomaly=anomalies[np.random.randint(0,anomalies.shape[0],1)[0],:]

    print('Computing KernelSHAP Local Importances (for a single anomaly)')
    print('#'*50)

    # Downsample the dataset to the background size
    dataset.downsample(max_samples=int(background*dataset.X.shape[0]))

    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

    if pre_process and scenario==2:
        dataset.initialize_test()
        dataset.pre_process()
    elif scenario==2 and not pre_process:
        dataset.initialize_test()
    elif scenario==1 and not pre_process:
        dataset.initialize_train_test()

    def EIF_score_function_shap(data):
        return I.predict(data)

    start_time = time.time()
    shap_explainer = shap.KernelExplainer(EIF_score_function_shap, dataset.X_test)
    shap_values = shap_explainer.shap_values(anomaly)
    shap_time = time.time()-start_time

    return shap_time

def compute_local_importances_kernelSHAP(I: Type[ExtendedIsolationForest],
                                 dataset: Type[Dataset],
                                 background:float=0.1,
                                 pre_process:float=False,
                                 scenario:int=2,
                                 n_anomalies:int=100) -> float: 
    
    """
    Compute the local importance score for a certain number of anomalies with KernelSHAP method 

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Type[Dataset]): Input dataset.
        background (float): The percentage of the dataset to use as background. Defaults to 0.1.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        pre_process (bool): Whether to pre process the dataset after computing the downsampled version according to the background. Defaults to False.
        scenario (int): The scenario of the experiment. Defaults to 2.
        n_anomalies (int): Number of anomalies on which to compute the importance scores 
    
    Returns:
        The time to compute the local feature importances for a single anomaly 
    """

    def EIF_score_function_shap(data):
        return I.predict(data)

    I.fit(dataset.X_train)

    # y_pred=I._predict(dataset.X_test,p).astype(int)
    # anomalies=dataset.X_test[np.where(y_pred==1)[0]]
    # anomaly=anomalies[np.random.randint(0,anomalies.shape[0],1)[0],:]

    print(f'Computing KernelSHAP importance scores for the {n_anomalies} most anomalous points')
    print('#'*50)

    # Downsample the dataset to the background size
    dataset.downsample(max_samples=int(background*dataset.X.shape[0]))

    dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

    if pre_process and scenario==2:
        dataset.initialize_test()
        dataset.pre_process()
    elif scenario==2 and not pre_process:
        dataset.initialize_test()
    elif scenario==1 and not pre_process:
        dataset.initialize_train_test()

    scores=EIF_score_function(dataset.X_test)
    # Find the n_anomalies most anomalous points
    anomalies_idx=np.argsort(scores)[:n_anomalies]
    anomalies=dataset.X_test[anomalies_idx]

    # Compute the shap_values for all the selected anomalies 
    imp_mat=np.zeros((n_anomalies,dataset.X_test.shape[1]))
    shap_explainer = shap.KernelExplainer(EIF_score_function_shap, dataset.X_test)
    for i,anomaly in enumerate(anomalies):
        imp_mat[i,:] = shap_explainer.shap_values(anomaly)

        if i%5==0:
            print("#"*50)
            print(f'Computed importance score of {i} anomalies ')
            print(imp_mat[i-5:i,:])
            print("#"*50)

    return imp_mat

def compute_plt_data(imp_path:str) -> dict:

    """
    Compute statistics on the global feature importances obtained from experiment_global_importances. These will then be used in the score_plot method. 

    Args:
        imp_path: The path to the importances file.
    
    Returns:
        The dictionary containing the mean importances, the feature order, and the standard deviation of the importances.
    """

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
                      random: bool = False,
                      scenario:int=2
                      ) -> np.array:
        
        """
        Perform feature selection on the dataset by dropping features in order of importance.

        Args:
            I: The AD model.
            dataset: Input dataset.
            importances_indexes: The indexes of the features in the dataset.
            n_runs: The number of runs. Defaults to 10.
            inverse: Whether to drop the features in decreasing order of importance. Defaults to True.
            random: Whether to drop the features in random order. Defaults to False.
            scenario: The scenario of the experiment. Defaults to 2.
        
        Returns:
            The average precision scores for the different runs.
        """

        dataset_shrinking = copy.deepcopy(dataset)
        d = dataset.X.shape[1]
        precisions = np.zeros(shape=(len(importances_indexes),n_runs))
        for number_of_features_dropped in tqdm(range(len(importances_indexes))):
            runs = np.zeros(n_runs)
            for run in range(n_runs):
                if random:
                    importances_indexes = np.random.choice(importances_indexes, len(importances_indexes), replace=False)
                dataset_shrinking.X = dataset.X_test[:,importances_indexes[:d-number_of_features_dropped]] if not inverse else dataset.X_test[:,importances_indexes[number_of_features_dropped:]]
                dataset_shrinking.y = dataset.y
                dataset_shrinking.drop_duplicates()
                
                if scenario==2:
                    dataset_shrinking.split_dataset(1-dataset_shrinking.perc_outliers,0)
                    dataset_shrinking.initialize_test()
                else:
                    dataset_shrinking.initialize_train()
                    dataset_shrinking.initialize_test()

                try:
                    if dataset.X.shape[1] == dataset_shrinking.X.shape[1]:
                        
                        start_time = time.time()
                        I.fit(dataset_shrinking.X_train)
                        fit_time = time.time() - start_time
                        
                        if run >3:
                            dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time)
                        start_time = time.time()
                        score = I.predict(dataset_shrinking.X_test)
                        predict_time = time.time() - start_time
                        
                        if run >3:                        
                            dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)
                    else:
                        I.fit(dataset_shrinking.X_train)
                        score = I.predict(dataset_shrinking.X_test)
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
                                                   interpretation:str="EXIFFI+",
                                                   pre_process:bool=True,
                                                   ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    
    """
    Evaluate the average precision of the model on the dataset for different contamination values in the training set. 
    The precision values will then be used in the `plot_precision_over_contamination` method

    Args:
        I: The AD model.
        dataset: Input dataset.
        n_runs: The number of runs. Defaults to 10.
        train_size: The size of the training set. Defaults to 0.8.
        contamination_values: The contamination values. Defaults to `np.linspace(0.0,0.1,10)`.
        compute_GFI: Whether to compute the global feature importances. Defaults to False.
        interpretation: Name of the interpretation method to be used. Defaults to "EXIFFI+".
        pre_process: Whether to pre process the dataset. Defaults to True.

    Returns:
        The average precision scores and the global feature importances if `compute_GFI` is True, 
        otherwise just the average precision scores are returned. 
    """

    precisions = np.zeros(shape=(len(contamination_values),n_runs))
    if compute_GFI:
        importances = np.zeros(shape=(len(contamination_values),n_runs,len(contamination_values),dataset.X.shape[1]))
    for i,contamination in tqdm(enumerate(contamination_values)):
        for j in range(n_runs):
            dataset.split_dataset(train_size,contamination)
            dataset.initialize_test()

            if pre_process:
                dataset.pre_process()
            
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
            score = I.predict(dataset.X_test)
            predict_time = time.time() - start_time
            if j>3:
                try:
                    dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)
                except:
                    print('Model not recognized: creating a new key in the dict_time for the new model')
                    dict_time["predict"].setdefault(I.name, {}).setdefault(dataset.name, []).append(predict_time)
            
            avg_prec = sklearn.metrics.average_precision_score(dataset.y_test,score)
            precisions[i,j] = avg_prec
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
    if compute_GFI:
        return precisions,importances
    return precisions

def performance(y_pred:np.array,
                y_true:np.array,
                score:np.array,
                I:Type[ExtendedIsolationForest],
                model_name:str,
                dataset:Type[Dataset],
                contamination:float=0.1,
                train_size:float=0.8,
                scenario:int=2,
                n_runs:int=10,
                filename:str="",
                path:str=os.getcwd(),
                save:bool=True
                ) -> tuple[pd.DataFrame,str]: 
    
    """
    Compute the performance metrics of the model on the dataset.

    Args:
        y_pred: The predicted labels.
        y_true: The true labels.
        score: The Anomaly Scores.
        I: The AD model.
        model_name: The name of the model.
        dataset: Input dataset.
        contamination: The contamination factor. Defaults to 0.1.
        train_size: The size of the training set. Defaults to 0.8.
        scenario: The scenario of the experiment. Defaults to 2.
        n_runs: The number of runs. Defaults to 10.
        filename: The filename. Defaults to "".
        path: The path to the experiments folder. Defaults to os.getcwd().
        save: Whether to save the results. Defaults to True.

    Returns:
        The performance metrics and the path to the results.
    """

    y_pred=y_pred.astype(int)
    y_true=y_true.astype(int)

    if dataset.X.shape[0]>7500:
        dataset.downsample(max_samples=7500)

    precisions=[]
    for i in trange(n_runs):
        I.fit(dataset.X_train)
        if model_name in ['DIF','AnomalyAutoencoder']:
            score = I.decision_function(dataset.X_test)
        else:
            score = I.predict(dataset.X_test)
        precisions.append(average_precision_score(y_true, score))
    
    df=pd.DataFrame({
        "Model": model_name,
        "Dataset": dataset.name,
        "Contamination": contamination,
        "Train Size": train_size,
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "f1 score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Average Precision": np.mean(precisions),
        "ROC AUC Score": roc_auc_score(y_true, y_pred)
    }, index=[pd.Timestamp.now()])

    path=path + f"/experiments/results/{dataset.name}/experiments/metrics/{model_name}/" + f"scenario_{str(scenario)}/"

    if not os.path.exists(path):
        os.makedirs(path)
    
    filename=f"perf_{dataset.name}_{model_name}_{scenario}"

    if save:
        save_element(df, path, filename)
    
    return df,path

def ablation_EIF_plus(I:Type[ExtendedIsolationForest], 
                      dataset:Type[Dataset], 
                      eta_list:list[float], 
                      nruns:int=10) -> list[np.array]:

    """
    Compute the average precision scores for different values of the eta parameter in the EIF+ model.

    Args:
        I: The AD model.
        dataset: Input dataset.
        eta_list: The list of eta values.
        nruns: The number of runs. Defaults to 10.

    Returns:
        The average precision scores.
    """

    precisions = []
    for eta in tqdm(eta_list):
        precision = []
        for run in range(nruns):
            I.eta = eta
            I.fit(dataset.X_train)
            score = I.predict(dataset.X_test)
            precision.append(average_precision_score(dataset.y_test, score))
        precisions.append(precision)
    return precisions
        
        
    


