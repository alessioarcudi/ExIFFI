from __future__ import annotations

from typing import Type, Optional, List
import numpy.typing as npt
from dataclasses import dataclass, field

from scipy.io import loadmat
import mat73
import os 
import json
import glob

import numpy as np
import random 
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.preprocessing import StandardScaler
import random
import copy


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler

@dataclass
class Dataset:
    """
    A class to represent a dataset.

    Attributes:
        name: The name of the dataset.
        path: The path to the dataset file.
        feature_names_filepath: The path to the json file containing the feature names of the dataset.
        X: Data matrix of the dataset.
        y: The labels of the dataset.
        X_train: Training set, initialized to None
        y_train: The labels of the training set
        X_test: Test set, initialized to None
        y_test: The labels of the test set
        feature_names: The names of the features of the dataset.
    """
    name: str
    path: str = "../data/"
    feature_names_filepath: Optional[str] = None
    X: Optional[npt.NDArray] = field(default=None, init=False)
    y: Optional[npt.NDArray] = field(default=None, init=False)
    X_train: Optional[npt.NDArray] = field(default=None, init=False)
    y_train: Optional[npt.NDArray] = field(default=None, init=False)
    X_test: Optional[npt.NDArray] = field(default=None, init=False)
    y_test: Optional[npt.NDArray] = field(default=None, init=False)
    feature_names: Optional[List[str]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize the dataset.

        Load the dataset from the file and set the feature names.
        
        """
        self.load()

        if self.feature_names_filepath is not None:
            self.dataset_feature_names()

        if self.feature_names is None:
            self.feature_names=np.arange(self.shape[1])
        
    @property
    def shape(self) -> tuple:
        """
        Return the shape of the dataset.

        Returns:
            The shape of the dataset.
        """
        return self.X.shape if self.X is not None else ()
    
    @property
    def n_outliers(self) -> int:
        """
        Return the number of outliers in the dataset.
        
        Returns:
            The number of outliers in the dataset.
        """
        return int(sum(self.y)) if self.y is not None else 0
    
    @property
    def perc_outliers(self) -> float:
        """
        Return the percentage of outliers in the dataset (i.e. the contamination factor)

        Returns:
            The percentage of outliers in the dataset.
        """
        return sum(self.y) / len(self.y) if self.y is not None else 0.0
        
    def load(self) -> None:
        """
        Load the dataset from the file.

        Raises:
            FileNotFoundError: If the dataset file is not found.
            Exception: If the dataset name is not valid.

        Returns:
            The dataset is loaded in place.
        """
        try:
            datapath=glob.glob(self.path+'/*.mat')
            index = [i for i, path in enumerate(datapath) if os.path.basename(path).startswith(self.name)]
            try:
                mat = loadmat(datapath[index[0]])
            except NotImplementedError:
                mat = mat73.loadmat(datapath[index[0]])
                
            self.X = mat['X'].astype(float)
            self.y = mat['y'].reshape(-1, 1).astype(float)
        except FileNotFoundError:
            try:
                datapath=glob.glob(self.path+'/*.csv')
                index = [i for i, path in enumerate(datapath) if os.path.basename(path).startswith(self.name)]
                T = pd.read_csv(datapath)
                if 'Unnamed: 0' in T.columns:
                    T = T.drop(columns=['Unnamed: 0'])
                self.X = T['X'].to_numpy(dtype=float)
                self.y = T['y'].to_numpy(dtype=float).reshape(-1, 1)
            except Exception as e:
                try:
                    datapath=glob.glob(self.path+'/*.csv')
                    index = [i for i, path in enumerate(datapath) if os.path.basename(path).startswith(self.name)]
                    if self.name == "glass":
                        T = pd.read_csv(datapath)
                    else:
                        T = pd.read_csv(datapath,index_col=0)
                    if 'Unnamed: 0' in T.columns:
                        T = T.drop(columns=['Unnamed: 0'])
                    self.X = T.loc[:,T.columns != "Target"].to_numpy(float)
                    self.y = T.loc[:,"Target"].to_numpy(float)
                except:
                    raise Exception("The dataset name is not valid") from e


    def __repr__(self) -> str:
        return f"[{self.name}][{self.shape}][{self.n_outliers}]"

    def drop_duplicates(self) -> None:
        """
        Drop duplicate samples from the dataset.

        Returns:
            The dataset is modified in place.
        """
        S = np.c_[self.X, self.y]
        S = pd.DataFrame(S).drop_duplicates().to_numpy()
        self.X, self.y = S[:, :-1], S[:, -1]
        
    def downsample(self, max_samples: int = 2500) -> None:
        """
        Downsample the dataset to a maximum number of samples.

        Args:
            max_samples: The maximum number of samples to keep in the dataset.
        
        Returns:
            The dataset is modified in place.
        """
        if len(self.X) > max_samples:
            print("downsampled to ", max_samples)
            sss = SSS(n_splits=1, test_size=1 - max_samples / len(self.X))
            index = list(sss.split(self.X, self.y))[0][0]
            self.X, self.y = self.X[index, :], self.y[index]
    
    def partition_data(self,X:np.array,y:np.array) -> tuple:

        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        try:
            inliers = X[y == 0, :]
            outliers = X[y == 1, :]
            y_inliers= y[y == 0]
            y_outliers= y[y == 1]
        except TypeError:
            print('X_train and y_train not loaded yet. Run split_dataset() first')
            return 
        return inliers, outliers,y_inliers,y_outliers
    
    def print_dataset_resume(self) -> None:
        """
        Print a summary of the dataset.

        The summary includes the number of samples, the number of features, the number of inliers and outliers and some
        summary statistics of the features.

        Returns:
            The dataset summary is printed.
        
        """
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return

        # Basic statistics
        num_samples = len(self.X)
        num_features = self.X.shape[1] if self.X is not None else 0
        num_inliers = np.sum(self.y == 0)
        num_outliers = np.sum(self.y == 1)
        balance_ratio = num_outliers / num_samples

        # Aggregate statistics for features in X
        mean_values = np.mean(self.X, axis=0)
        std_dev_values = np.std(self.X, axis=0)
        min_values = np.min(self.X, axis=0)
        max_values = np.max(self.X, axis=0)

        # Compact representation of statistics
        mean_val = np.mean(mean_values)
        std_dev_val = np.mean(std_dev_values)
        min_val = np.min(min_values)
        max_val = np.max(max_values)

        # Print the summary
        print(f"Dataset Summary for '{self.name}':")
        print(f" Total Samples: {num_samples}, Features: {num_features}")
        print(f" Inliers: {num_inliers}, Outliers: {num_outliers}, Balance Ratio: {balance_ratio:.2f}")
        print(f" Feature Stats - Mean: {mean_val:.2f}, Std Dev: {std_dev_val:.2f}, Min: {min_val}, Max: {max_val}")


    def split_dataset(self, 
                      train_size:float = 0.8, 
                      contamination:float = 0.1) -> None:
        
        """
        Split the dataset into training and test sets with a given train size and contamination factor.

        Args:
            train_size: The proportion of the dataset to include in the training set.
            contamination: The proportion of outliers in the dataset.

        Returns:
            The dataset is split into training and test sets in place

        """
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        
        # Check if train_size is correct
        if train_size > 1 - self.perc_outliers:
            print("Train size is too large. Setting it at 1-dataset.perc_outliers.")
            train_size = 1 - self.perc_outliers
        
        indexes_outliers = np.where(self.y==1)[0].tolist()
        indexes_inliers = np.where(self.y==0)[0].tolist()
        random.shuffle(indexes_outliers)
        random.shuffle(indexes_inliers)
        dim_train = int(len(self.X)*train_size)
        self.X_train = np.zeros((dim_train,self.X.shape[1]))
        self.y_train = np.zeros(dim_train)
        for i in range(dim_train):
            if i < dim_train*contamination and len(indexes_outliers) > 0:
                index = indexes_outliers.pop()
            else:
                index = indexes_inliers.pop()
            self.X_train[i] = self.X[index]
            self.y_train[i] = self.y[index]

    def pre_process(self) -> None:

        """
        Normalize the data using `StansardScaler()` from `sklearn.preprocessing`.

        Returns:
           The dataset is normalized in place.
        """
        
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        if self.X_train is None:
            self.initialize_train_test()
        if self.X_test is None:
            self.initialize_test()

        scaler = StandardScaler()
        
        self.X_train=scaler.fit_transform(self.X_train)
        self.X_test=scaler.transform(self.X_test)

    def initialize_train_test(self) -> None:

        """
        Initialize the training and test sets with the original dataset. 

        This method is used when `split_dataset()` has not been called before `pre_process()`.

        Returns:
            The training and test sets are initialized in place.
        """
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        if self.X_train is None:
            self.initialize_train()
        if self.X_test is None:
            self.initialize_test()

    def initialize_test(self) ->None:

        """
        Initialize the test set with the original dataset. 

        This method is used when `split_dataset()` has not been called before `pre_process()`.

        Returns:
            The test set is initialized in place.
        """

        self.X_test=copy.deepcopy(self.X)
        self.y_test=copy.deepcopy(self.y)

    
    def initialize_train(self) ->None:

        """
        Initialize the train set with the original dataset. 

        This method is used when `split_dataset()` has not been called before `pre_process()`.

        Returns:
            The training set is initalized in place.
        """

        self.X_train=copy.deepcopy(self.X)
        self.y_train=copy.deepcopy(self.y)

    def dataset_feature_names(self) -> List[str]:

            """ 
            Set the feture names for the datasets for which the feature names are available 

            Returns:
                Set the feature_names attributes to a list of string containing the feature names of the dataset.
            """
            with open(self.feature_names_filepath+'data_feature_names.json','r') as f:
                data_feature_names=json.load(f)

            if self.name in data_feature_names:    
                self.feature_names=data_feature_names[self.name]
            else:
                self.feature_names=None 
        

 
