import time
from typing import Type,Union
import pickle
import numpy as np
import pandas as pd
import os
from collections import namedtuple
from model_reboot.EIF_reboot import ExtendedIsolationForest
from utils_reboot.datasets import Dataset
from sklearn.ensemble import IsolationForest 
from pyod.models.dif import DIF as oldDIF
from pyod.models.auto_encoder import AutoEncoder as oldAutoEncoder
from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score

Precisions = namedtuple("Precisions",["direct","inverse","dataset","model","value"])
NewPrecisions = namedtuple("NewPrecisions", ["direct", "inverse", "dataset", "model", "value", "aucfs"])
Precisions_random = namedtuple("Precisions_random",["random","dataset","model"])


class sklearn_IsolationForest(IsolationForest):
    
    """
    Wrapper of `sklearn.ensemble.IsolationForest` 
    """

    def __init__(self, **kwargs):

        """
        Constructor of the class `sklearn_IsolationForest` which uses the constructor of the parent class `IsolationForest` from `sklearn.ensemble` module.

        Attributes:
            name (str): Add the name attribute to the class.
        """
        super().__init__(**kwargs)
        self.name = "sklearn_IF"
    
    def predict(self, X:np.array) -> np.array:

        """
        Overwrite the `predict` method of the parent class `IsolationForest` from `sklearn.ensemble` module to obtain the 
        Anomaly Scores instead of the class labels (i.e. inliers and outliers)

        Args:
            X: Input dataset

        Returns:
            Anomaly Scores 
        """

        score=self.decision_function(X)
        return -1*score+0.5

class DIF(oldDIF):

    """
    Wrapper of `pyod.models.dif.DIF`
    """

    def __init__(self, **kwargs):

        """
        Constructor of the class `DIF` which uses the constructor of the parent class `DIF` from `pyod.models.dif` module.
        
        Attributes:
            name (str): Add the name attribute to the class.
        """
        super().__init__(**kwargs)
        self.name = "DIF"
    
    def predict(self, X:np.array) -> np.array:

        """
        Overwrite the `predict` method of the parent class `DIF` from `pyod.models.dif` module to obtain the
        Anomaly Scores instead of the class labels (i.e. inliers and outliers)

        Args:
            X: Input dataset

        Returns:
            Anomaly Scores 

        """

        score=self.decision_function(X)
        return score
    
    def _predict(self,
                 X:np.array,
                 p:float)->np.array:

        """
        Method to predict the class labels based on the Anomaly Scores and the contamination factor `p`

        Args:
            X: Input dataset
            p: Contamination factor

        Returns:
            Class labels (i.e. 0 for inliers and 1 for outliers)
        """

        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat

    
class AutoEncoder(oldAutoEncoder):

    """
    Wrapper of `pyod.models.auto_encoder.AutoEncoder`
    """

    def __init__(self, **kwargs):

        """
        Constructor of the class `AutoEncoder` which uses the constructor of the parent class `AutoEncoder` from `pyod.models.auto_encoder` module.

        Attributes:
            name (str): Add the name attribute to the class.
        """

        super().__init__(**kwargs)
        self.name = "AnomalyAutoencoder"
    
    def predict(self, X:np.array) -> np.array:

        """
        Overwrite the `predict` method of the parent class `AutoEncoder` from `pyod.models.auto_encoder` module to obtain the
        Anomaly Scores instead of the class labels (i.e. inliers and outliers)

        Args:
            X: Input dataset

        Returns:
            Anomaly Scores
        """
        score=self.decision_function(X)
        return score
    
    def _predict(self,
                 X:np.array,
                 p:float)-> np.array:

        """
        Method to predict the class labels based on the Anomaly Scores and the contamination factor `p`

        Args:
            X: Input dataset
            p: Contamination factor

        Returns:
            Class labels (i.e. 0 for inliers and 1 for outliers)
        """

        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat
    
def get_feature_indexes(dataset:Type[Dataset],
                        f1:Union[str, int],
                        f2:Union[str, int]) -> tuple[int,int]:
    
    """
    Function to get the indexes of two features in the dataset given the feature names. 

    Args:
        dataset: Dataset
        f1: Name of the first feature
        f2: Name of the second feature
    
    Returns:
        Indexes of the two features in the dataset
    """

    if isinstance(f1,int) and isinstance(f2,int):
        return f1,f2

    feature_names=dataset.feature_names

    try:
        idx1=feature_names.index(f1)
    except:
        print('Feature name not valid')
    try: 
        idx2=feature_names.index(f2)
    except:
        print('Feature name not valid')

    return idx1,idx2
    

def save_element(element:Union[np.array,list,pd.DataFrame,Type[Precisions],Type[NewPrecisions],Type[Precisions_random]],
                 directory_path:str,
                 filename:str="",
                 filetype:str="pickle",
                 no_time:str=False) -> None:
    
    """
    Function to save an element produced by an experiment in a file (i.e. `npz` or `pickle` file) in the specified directory path.

    Args:
        element: Element to be saved
        directory_path: Directory path where the file will be saved
        filename: Name of the file
        filetype: Type of the file (i.e. `npz` or `pickle`)
        no_time: Boolean value to indicate whether the time should be included in the filename or not, by default False

    Returns:
        The method saves element and does not return any value 

    """

    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    if no_time == False:
        filename = current_time + '_' + filename
    path = directory_path + '/' + filename
    if filetype == "pickle":
        with open(path+".pickle", 'wb') as fl:
            pickle.dump(element, fl)
    elif filetype == "npz":
        np.savez(path, element=element)
        
def get_most_recent_file(directory_path:str,
                         filetype:str="pickle",
                         file_pos:int=0)->str:

    """
    Function to get the most recent file (i.e. last modified file) in a directory path.

    Args:
        directory_path: Directory path where the files are stored
        filetype: Type of the file (i.e. `npz` or `pickle`)
        file_pos: Position of the file in the sorted list of files (i.e. 0 for the most recent file, 1 for the second most recent file, etc.)

    Returns:
        Path to the most recent file in the directory path

    """
    
    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    date_format = "%d-%m-%Y_%H-%M-%S"
    datetimes=[datetime.strptime(file[:19],date_format) for file in os.listdir(directory_path)]
    sorted_files=sorted(datetimes,reverse=True)
    most_recent_file=sorted_files[file_pos].strftime(date_format)+f'_.{filetype}'
    return os.path.join(directory_path,most_recent_file)

def open_element(file_path:str,
                 filetype:str="pickle") -> Union[np.array,list,pd.DataFrame,Type[Precisions],Type[NewPrecisions],Type[Precisions_random]]:

    """
    Function to open an element from a file (i.e. `npz` or `pickle` file) in the specified directory path.

    Args:
        file_path: Path to the file
        filetype: Type of the file (i.e. `npz` or `pickle`)

    Returns:
        Element stored in the file
    """

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

def save_fs_prec(precs:namedtuple,
                 path:str) -> None:

    """
    Function to save the feature selection precisions in a file (i.e. `pickle` file) in the specified directory path.

    Args:
        precs: Feature selection precisions
        path: Directory path where the file will be saved

    Returns:
        The method saves the feature selection precisions and does not return any value

    """

    #aucfs=sum(precs.inverse.mean(axis=1)-precs.direct.mean(axis=1))
    aucfs=np.nansum(np.nanmean(precs.inverse,axis=1)-np.nanmean(precs.direct,axis=1))
    new_precs = NewPrecisions(direct=precs.direct,
                            inverse=precs.inverse,
                            dataset=precs.dataset,
                            model=precs.model,
                            value=precs.value,
                            aucfs=aucfs)
    save_element(new_precs, path, filetype="pickle")

def save_fs_prec_random(precs:namedtuple,
                        path:str) -> None:
    
    """
    Function to save the feature selection precisions for random features in a file (i.e. `pickle` file) in the specified directory path.

    Args:
        precs: Feature selection precisions for random features
        path: Directory path where the file will be saved

    Returns:
        The method saves the feature selection precisions for random features and does not return any value
    """

    new_precs = Precisions_random(random=precs.random,
                            dataset=precs.dataset,
                            model=precs.model)
    save_element(new_precs, path, filetype="pickle")

def get_fs_file(dataset,model,interpretation,scenario):
    path=os.path.join(os.getcwd(),dataset.name,'experiments','feature_selection',model,interpretation,f'scenario_{str(scenario)}')
    file_path=get_most_recent_file(path)
    precs=open_element(file_path)
    return precs

def select_scenario() -> int:

    """
    Function to select the scenario for the experiment (i.e. Scenario 1 or Scenario 2) asking the user to input the scenario number.

    This method was specifically designed to construct the `tutorial.ipynb` notebook for the documentation.

    Returns:
        The selected scenario number (i.e. 1 for Scenario 1 and 2 for Scenario 2)
    """

    scenario=int(input("Press 1 for scenario 1 and 2 for scenario 2: "))
    assert scenario in [1,2], "Scenario not recognized: Accepted values: [1,2]"
    return scenario

def select_pre_process() -> bool:

    """
    Function to select the pre-processing of the dataset asking the user to input the pre-processing number.

    This method was specifically designed to construct the `tutorial.ipynb` notebook for the documentation.

    Returns:
        Boolean value to indicate whether the dataset should be pre-processed or not 
        (i.e. 1 to pre-process the dataset and 2 otherwise)
    """
    pre_process=int(input("Press 1 to pre process the dataset, 2 otherwise: "))
    assert pre_process in [1,2], "Input values not recognized: Accepted values: [1,2]"
    return pre_process==1

def select_pre_process_scenario(dataset:Type[Dataset]) -> int:

    """
    Combine the selection of the pre-processing of the dataset and the scenario for the experiment.

    Args:
        dataset: Dataset to be used in the experiment

    Returns:
        The selected scenario number (i.e. 1 for Scenario 1 and 2 for Scenario 2)
    """
    pre_process=select_pre_process()
    scenario=select_scenario()

    if scenario==2:
        dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=0)

    if pre_process==1:
        dataset.pre_process()
        print("Dataset pre processed\n")
    elif scenario==2 and not pre_process==2:
        print("Dataset not preprocessed\n")
        dataset.initialize_test()
    elif scenario==1 and not pre_process==2:
        print("Dataset not preprocessed\n")
        dataset.initialize_train_test()

    print(f'Scenario: {scenario}\n')

    print(f'X_train shape: {dataset.X_train.shape}')
    print(f'X_test shape: {dataset.X_test.shape}')

    return scenario 