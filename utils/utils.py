import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from numba import jit

import sys;
#sys.path.append("..//models")
from models.interpretability_module import diffi_ib
#from utils.simulation_setup import MatFileDataset
from simulation_setup import MatFileDataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


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

#Remove?

def mean_confidence_interval_importances(l, confidence=0.95):
    """
    Mean value and confidence interval of a list of lists of results
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    l :         list
        list of lists of scores.

    Returns
    -------
    M:      list of tuples (m,m-h,m+h) where h is confidence interval
        
    """
    M=[]
    for i in range(l.shape[0]):
        a = 1.0 * np.array(l[i,:])
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        M.append((m, m-h, m+h))
    return M

#Remove?

def extract_order(X):
    X=X.sort_values(by=[0])
    X.reset_index(inplace=True)
    X.rename(columns={"index":"feature"},inplace=True)
    X.drop(labels=0,axis=1,inplace=True)
    X=X.squeeze()
    X.index=(X.index+1)*np.linspace(0,1,len(X))
    X=X.sort_values()
    return X.index

class MatFileDataset:

    def __init__(self):
        self.X = None
        self.y = None
        self.shape = None
        self.datasets = data

    def load(self, name: str):
        self.name = name
        try:
            mat = loadmat(self.name)
        except NotImplementedError:
            mat = mat73.loadmat(self.name)

        self.X = mat['X']
        self.y = mat['y'].reshape(-1, 1)
        self.shape = self.X.shape
        self.perc_anomalies = float(sum(self.y) / len(self.y))
        self.n_outliers = sum(self.y)

        
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

def dataset(name, path = "../data/"):
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
        datapath = path + name + ".mat"
    except FileNotFoundError:
        datapath = path + name + ".csv"

    if datapath[-3:]=="mat":
        T=MatFileDataset() 
        T.load(datapath)
    elif datapath[-3:]=="csv":
        T=pd.DataFrame()
        T["X"]=pd.read_csv(datapath)
    else:
        raise Exception("Sorry, the path is not valid")
    
    X,y = drop_duplicates(T.X,T.y)
    print(name, "\n")
    print_dataset_resume(X,y)
    
    return X,y

def csv_dataset(name, path = "../data/"):
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
    datapath = path + name + ".csv"
    data=pd.read_csv(datapath,index_col=0)
    if 'Unnamed: 0' in data.columns:
        data=data.drop(columns=['Unnamed: 0'])
    
    X=data[data.columns[data.columns!='Target']]
    y=data['Target']
    
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
    print("[numero elementi = {}]\n[percentage outliers = {}]\n[number features = {}]\n[number outliers = {}]".format(n_sample,perc_outliers,size,n_outliers))

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


''' 
def get_extended_test(n_pts,n_anomalies,cluster_distance,n_dim,anomalous_dim = [0]):
    #not_anomalous_dim     = np.setdiff1d(np.arange(n_dim),anomalous_dim)
    X1 = np.random.randn(n_pts,n_dim) + cluster_distance
    y1 = np.zeros(n_pts)
    #The first half of the additional anomalies are obtained subtracting 2*cluster_distance from the original points (X1)
    y1[:n_anomalies-int(n_anomalies/2)] = 1
    X1[:n_anomalies-int(n_anomalies/2),anomalous_dim] -= 2*cluster_distance
    #The second half of the additional anomalies are obtained adding 2*cluster_distance from the original points (X2)
    X2 = np.random.randn(n_pts,n_dim) - cluster_distance
    y2 = np.zeros(n_pts)
    y2[:int(n_anomalies/2)] = 1
    X2[:int(n_anomalies/2),anomalous_dim] += 2*cluster_distance
    #Concatenate together X1,X2 and y1,y2
    X = np.vstack([X1,X2])
    y = np.hstack([y1,y2])
    return X,y
'''

''' 
def cosine_ordered_similarity(v,u):
    v=np.exp((np.array(v)/np.array(v).max()))-1/(np.exp(1)-1)
    u=np.exp((np.array(u)/np.array(u).max()))-1/(np.exp(1)-1)
    S=1-(v).dot(u)/(np.sqrt(v.dot(v)*u.dot(u)))
    return S
'''
        