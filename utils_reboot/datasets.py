from __future__ import annotations

from typing import Type, Optional, List
import numpy.typing as npt
from dataclasses import dataclass, field

from scipy.io import loadmat
import mat73

import numpy as np
import random 
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit as SSS
import random


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler


def Dataset_feature_names(name:str):

    data_feature_names={
        'pima': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'moodify': ['duration (ms)', 'danceability', 'energy', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'spec_rate'],
       'diabetes': ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    }

    if name in data_feature_names:    
        return data_feature_names[name]
    else:
        return None 

@dataclass
class Dataset:
    """
    A class to represent a dataset.
    INSTANCE VARIABLES:
        name: str
            The name of the dataset.
        path: str
            The path to the dataset file.
        X: Optional[npt.NDArray]
            The input features of the dataset.
        y: Optional[npt.NDArray]
            The labels of the dataset.
        shape: tuple
            The shape of the dataset.
        n_outliers: int
            The number of outliers in the dataset.
    """
    name: str
    path: str = "../data/"
    X: Optional[npt.NDArray] = field(default=None, init=False)
    y: Optional[npt.NDArray] = field(default=None, init=False)
    X_train: Optional[npt.NDArray] = field(default=None, init=False)
    y_train: Optional[npt.NDArray] = field(default=None, init=False)
    feature_names: Optional[List[str]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.load()
        self.feature_names=Dataset_feature_names(self.name)
        if self.feature_names is None:
            self.feature_names=np.arange(self.shape[1])
        
    @property
    def shape(self) -> tuple:
        return self.X.shape if self.X is not None else ()
    
    @property
    def n_outliers(self) -> int:
        return int(sum(self.y)) if self.y is not None else 0
    
    @property
    def perc_outliers(self) -> float:
        return sum(self.y) / len(self.y) if self.y is not None else 0.0
        
    def load(self) -> None:
        try:
            datapath = self.path + self.name + ".mat"
            try:
                mat = loadmat(datapath)
            except NotImplementedError:
                mat = mat73.loadmat(datapath)
                
            self.X = mat['X'].astype(float)
            self.y = mat['y'].reshape(-1, 1).astype(float)
        except FileNotFoundError:
            try:
                datapath = self.path + self.name + ".csv"
                T = pd.read_csv(datapath)
                self.X = T['X'].to_numpy(dtype=float)
                self.y = T['y'].to_numpy(dtype=float).reshape(-1, 1)
            except Exception as e:
                try:
                    datapath = self.path + self.name + ".csv"
                    T = pd.read_csv(datapath,index_col=0)
                    self.X = T.loc[:,T.columns != "Target"].to_numpy(float)
                    self.y = T.loc[:,"Target"].to_numpy(float)
                except:
                    raise Exception("The dataset name is not valid") from e

    def __repr__(self) -> str:
        return f"[{self.name}][{self.shape}][{self.n_outliers}]"

    def drop_duplicates(self) -> None:
        S = np.c_[self.X, self.y]
        S = pd.DataFrame(S).drop_duplicates().to_numpy()
        self.X, self.y = S[:, :-1], S[:, -1]
        
    def downsample(self, max_samples: int = 2500) -> None:
        if len(self.X) > max_samples:
            print("downsampled to ", max_samples)
            sss = SSS(n_splits=1, test_size=1 - max_samples / len(self.X))
            index = list(sss.split(self.X, self.y))[0][0]
            self.X, self.y = self.X[index, :], self.y[index]
    
    def partition_data(self,X,y) -> tuple:

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
    
    def split_dataset(self, train_size = 0.8, contamination: float = 0.1) -> tuple:
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        
        inexes_outliers = np.where(self.y==1)[0].tolist()
        indexes_inliers = np.where(self.y==0)[0].tolist()
        random.shuffle(inexes_outliers)
        random.shuffle(indexes_inliers)
        dim_train = int(len(self.X)*train_size)
        self.X_train = np.zeros((dim_train,self.X.shape[1]))
        self.y_train = np.zeros(dim_train)
        for i in range(dim_train):
            if i < dim_train*contamination and len(inexes_outliers) > 0:
                index = inexes_outliers.pop()
            else:
                index = indexes_inliers.pop()
            self.X_train[i] = self.X[index]
            self.y_train[i] = self.y[index]

    def split_dataset(self, train_size = 0.8, contamination: float = 0.1) -> tuple:
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        
        inexes_outliers = np.where(self.y==1)[0].tolist()
        indexes_inliers = np.where(self.y==0)[0].tolist()
        random.shuffle(inexes_outliers)
        random.shuffle(indexes_inliers)
        dim_train = int(len(self.X)*train_size)
        self.X_train = np.zeros((dim_train,self.X.shape[1]))
        self.y_train = np.zeros(dim_train)
        for i in range(dim_train):
            if i < dim_train*contamination and len(inexes_outliers) > 0:
                index = inexes_outliers.pop()
            else:
                index = indexes_inliers.pop()
            self.X_train[i] = self.X[index]
            self.y_train[i] = self.y[index]
    
    def pre_process(self,
                    X_train: np.array,
                    X_test: np.array,
                    scaler:Optional[Type[StandardScaler]]=StandardScaler(),
                    split:bool=True) -> tuple:
    
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return

        
        if split:
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)
            X=np.r_[X_train,X_test]
            y_train=np.zeros(X_train.shape[0])
            y_test=np.ones(X_test.shape[0])
            y=np.concatenate([y_train,y_test])
            return X_train,X_test,X,y
        elif split==False:
            #Ensure X_train is not None
            if self.X_train is None:
                print("X_train not loaded. Load it running split_dataset() first")
                return
            self.X = scaler.fit_transform(self.X)
            self.X_train = scaler.fit_transform(self.X_train)

            

