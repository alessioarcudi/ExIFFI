import numpy as np
import pandas as pd
import scipy
import os
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from numba import jit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

"""
Utility Functions for models 
"""

def make_rand_vector(df,dimensions):
    """
    Random unitary vector in the unit ball with max number of dimensions
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    df: Degrees of freedom

    dimensions: number of dimensions of the feature space

    If df<dimensions then dimensions-df indexes will be set to 0. 

    Returns
    ----------
    n:      random vector: the normal to the splitting hyperplane
        
    """
    if dimensions<df:
        raise ValueError("degree of freedom does not match with dataset dimensions")
    else:
        vec_ = np.random.normal(loc=0.0, scale=1.0, size=df)
        indexes = np.random.choice(range(dimensions),df,replace=False)
        vec = np.zeros(dimensions)
        vec[indexes] = vec_
        vec=vec/np.linalg.norm(vec)
    return vec



@jit(nopython=True) 
def c_factor(n):
    """
    Average path length of unsuccesful search in a binary search tree given n points
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    n :         int
        Number of data points for the BST.

    Returns
    -------
    float:      Average path length of unsuccesful search in a BST
        
    """
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

"""
Utility Functions for experiments 
"""

def get_filename_suffix(dataset_names):
    suffix=''
    for data in dataset_names:
        suffix+=data[0]+'_'

    return suffix

def get_scaler(scaler_name="StandardScaler"):
    if scaler_name == "StandardScaler":
        return StandardScaler()
    elif scaler_name == "MinMaxScaler":
        return MinMaxScaler()
    elif scaler_name == "MaxAbsScaler":
        return MaxAbsScaler()
    elif scaler_name == "RobustScaler":
        return RobustScaler()
    else:
        raise ValueError(f"Scaler {scaler_name} not supported")
    
def set_p_distribution(model,distribution_name='normal_mean'):
        model.set_distribution(distribution_name)

def set_p_eta(model,eta=2):
        model.set_eta(eta)

def pre_process(scaler_name,X_train,X_test):
    """
    Pre processing function on the training and test set applying data normalization
    --------------------------------------------------------------------------------

    Parameters
    ----------
    scaler_name: str
        Name of the Scaler to use 
    X_train: pd.DataFrame
        Training Set
    X_test: pd.DataFrame
        Test Set

    Returns:
    -------
    X_train,X_test,X,y: Normalized version of the Training and Test Set, normalized version of the 
                        concatenation of X_train and X_test and testing labels 
    """

    X=np.r_[X_train,X_test]
    scaler=get_scaler(scaler_name)
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.fit_transform(X_test)
    y_train=np.zeros(X_train.shape[0])
    y_test=np.ones(X_test.shape[0])
    y=np.concatenate([y_train,y_test])
    X_test=np.r_[X_train,X_test]
    scaler2=get_scaler(scaler_name)

    """
    This X here will have the same shape as X_test but it is obtained scaling X_train and X_test 
    considered together
    """

    X=scaler2.fit_transform(X)
    return X_train,X_test,X,y


def load_preprocess(scaler,name:str,path:str):
    """
    Function loading the data from a file and pre processing it with the pre_process function 
    --------------------------------------------------------------------------------

    Parameters
    ----------
    scaler: str
        Name of the type of scaler to use to pre_process the data 
    name: str
        Dataset name
    path: str
        Dataset path 

    Returns:
    -------
    X_train,X_test,X,y: Normalized version of the Training and Test Set, normalized version of the 
                        concatenation of X_train and X_test and testing labels
    """
    extension = os.path.splitext(path)[1]

    if extension == ".csv":
        X, y = csv_dataset(name,path)
    elif extension == ".mat":
        print(f'Loading {name} dataset from {path}')
        X, y = mat_dataset(name,path)
    else:
        raise ValueError("Extension not supported")
    
    X_train,X_test=partition_data(X,y)
    X_train,X_test,X,y=pre_process(scaler,X_train,X_test)

    return X_train,X_test,X,y

def drop_duplicates(X,y):
    """
    Drop duplicate rows from a dataset
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    X :         pd.DataFrame
        Input dataset
    y:          np.array
        Dataset labels

    Returns
    -------
    X,y:      Updated dataset and labels after the duplicates removal
        
    """
    S=np.c_[X,y]
    S=pd.DataFrame(S).drop_duplicates().to_numpy()
    X,y = S[:,:-1], S[:,-1]
    return X,y

def mat_dataset(name,path):
    """
    Upload a dataset from a .mat file 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    name :         string
        Dataset's name
    path:          string
        Path of the .mat file containing the dataset

    Returns
    -------
    X,y:      X contains the dataset input features as a pd.DataFrame while y contains the dataset's labels as a np.array
        
    """

    try:
        data=loadmat(path)
    except FileNotFoundError:
        print('Wrong path or file extension')

    X,y=data['X'],data['y']
    X,y=drop_duplicates(X,y)

    print(name, "\n")
    print_dataset_resume(X,y)

    return X,y 


def csv_dataset(name,path):
    """
    Upload a dataset from a .csv file. This function was used for the Diabetes and Moodify datasets. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    name :         string
        Dataset's name
    path:          string
        Path of the .csv file containing the dataset

    Returns
    -------
    X,y:      X contains the dataset input features as a pd.DataFrame while y contains the dataset's labels as a np.array
        
    """

    data=pd.read_csv(path,index_col=0)
    if 'Unnamed: 0' in data.columns:
        data=data.drop(columns=['Unnamed: 0'])
    
    if 'Target' in data.columns:
        X=data[data.columns[data.columns!='Target']]
        y=data['Target']
    elif 'Outcome' in data.columns:
        X=data[data.columns[data.columns!='Outcome']]
        y=data['Outcome']

    X,y = drop_duplicates(X,y)
    print(name, "\n")
    print_dataset_resume(X,y)
    
    return X,y

def print_dataset_resume(X,y):
    """
    Print some useful information about a dataset loaded using the dataset(name,path) function 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    X :         pd.DataFrame
        Input dataset
    y:          np.array
        Dataset's labels

    Returns
    -------
    Print a message to the screen with some information about the dataset: number of elements, contamination factor, number of features and number of outliers
        
    """
    n_sample=int(X.shape[0])
    perc_outliers=sum(y)/X.shape[0]
    size=int(X.shape[1])
    n_outliers=int(sum(y))
    print("[number of samples = {}]\n[percentage outliers = {}]\n[number features = {}]\n[number outliers = {}]".format(n_sample,perc_outliers,size,n_outliers))

def downsample(X,y):
    """
    Downsample a dataset in case it contains more than 2500 samples 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    X :         pd.DataFrame
        Input dataset
    y:          np.array
        Dataset's labels

    Returns
    -------
    X,y: Downsamples dataset and labels 
        
    """
    if len(X)>2500:
        print("downsampled to 2500")
        sss = SSS(n_splits=1,test_size=1-2500/len(X))
        index = list(sss.split(X,y))[0][0]
        X,y = X[index,:],y[index]
        print(X.shape)
    return X,y

def partition_data(X,y):
    """
    Partition a dataset in the set of inliers and the set of outliers according to the dataset labels. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    X :         pd.DataFrame
        Input dataset
    y:          np.array
        Dataset's labels

    Returns
    -------
    inliers,outliers: Returns two subsets of the input dataset X: one containing the inliers, the other containing the outliers. 
        
    """
    inliers=X[y==0,:]
    outliers=X[y==1,:]
    return inliers,outliers
        