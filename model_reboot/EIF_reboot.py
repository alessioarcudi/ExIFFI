from __future__ import annotations

from typing import ClassVar, Optional, List
import numpy.typing as npt
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange, float64, int64, boolean
from numba.experimental import jitclass
from joblib import Parallel, delayed


from numba import njit
from numba.typed import List
import numpy as np
# import dask
# from dask.distributed import Client
from tqdm import tqdm

@njit
def make_rand_vector(df,dimensions) -> npt.NDArray[np.float64]:
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
        indexes = np.random.choice(np.arange(dimensions),df,replace=False)
        vec = np.zeros(dimensions)
        vec[indexes] = vec_
        vec=vec/np.linalg.norm(vec)
    return vec



@njit
def c_factor(n: int) :
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
    if n <=1: return 0
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


@njit    
def get_leaf_ids(X, child_left, child_right, normals, intercepts):
        res = []
        for x in X:
            node_id = 0
            while child_left[node_id] or child_right[node_id]:
                d = np.dot(np.ascontiguousarray(x),np.ascontiguousarray(normals[node_id]))
                node_id = child_left[node_id] if d <= intercepts[node_id] else child_right[node_id]        
            res.append(int(node_id))
        return np.array(res)

@njit
def calculate_importances(paths: np.ndarray, directions: np.ndarray, importances_left: np.ndarray, importances_right: np.ndarray, normals: np.ndarray, d: int):
    # Flatten the paths and directions for 1D boolean indexing
    paths_flat = paths.flatten()
    directions_flat = directions.flatten()
    
    # Create masks for left and right directions
    left_mask_flat = directions_flat == -1
    right_mask_flat = directions_flat == 1
    
    # Use masks to filter flattened paths; initialize with -1 (or suitable default)
    left_paths_flat = np.full_like(paths_flat, -1)
    right_paths_flat = np.full_like(paths_flat, -1)
    
    # Apply the masks
    left_paths_flat[left_mask_flat] = paths_flat[left_mask_flat]
    right_paths_flat[right_mask_flat] = paths_flat[right_mask_flat]
    
    # Since importances are mentioned to be arrays of arrays, let's assume we can index them directly with the flattened paths
    # Note: This step might need adjustment based on the actual structure and intended calculations
    importances_sum_left = np.zeros((paths.shape[0],d), dtype=np.float64)  # Initialize to match number of rows in paths
    importances_sum_right = np.zeros((paths.shape[0],d), dtype=np.float64)
    
    normals_sum = np.zeros((paths.shape[0],d), dtype=np.float64)  # Initialize to match number of rows in paths
    
    importances_sum_left = importances_left[left_paths_flat].reshape(paths.shape[0],paths.shape[1],d).sum(axis=1)
    importances_sum_right = importances_right[right_paths_flat].reshape(paths.shape[0],paths.shape[1],d).sum(axis=1)
    normals_sum = np.abs(normals[paths_flat]).reshape(paths.shape[0],paths.shape[1],d).sum(axis=1)

    importances_sum = importances_sum_left + importances_sum_right
    
    return importances_sum, normals_sum


tree_spec = [
    ("plus", boolean),
    ("locked_dims", int64),
    ("max_depth", int64),
    ("min_sample", int64),
    ("n", int64),
    ("d", int64),
    ("node_count", int64),
    ("path_to", int64[:, :]),
    ("path_to_Right_Left", int64[:, :]) ,    
    ("child_left", int64[:]),
    ("child_right", int64[:]),
    ("normals", float64[:, :]),
    ("intercepts", float64[:]),
    ("node_size", int64[:]),      
    ("depth", int64[:]),          
    ("corrected_depth", float64[:]),
    ("importances_right", float64[:, :]),
    ("importances_left", float64[:, :])
]

@jitclass(tree_spec)
class ExtendedTree:
    def __init__(self, n, d, max_depth, locked_dims=0, min_sample=1, plus=True, max_nodes=10000):
        self.plus = plus
        self.locked_dims = locked_dims 
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.n = n
        self.d = d
        self.node_count = 1

        self.path_to = -np.ones((max_nodes, max_depth+1), dtype=np.int64)
        self.path_to_Right_Left = np.zeros((max_nodes, max_depth+1), dtype=np.int64)
        self.child_left = np.zeros(max_nodes, dtype=np.int64)
        self.child_right = np.zeros(max_nodes, dtype=np.int64)
        self.normals = np.zeros((max_nodes, d), dtype=np.float64)
        self.intercepts = np.zeros(max_nodes, dtype=np.float64)
        self.node_size = np.zeros(max_nodes, dtype=np.int64)
        self.depth = np.zeros(max_nodes, dtype=np.int64)
        self.corrected_depth = np.zeros(max_nodes, dtype=np.float64)
        self.importances_right = np.zeros((max_nodes, d), dtype=np.float64)
        self.importances_left = np.zeros((max_nodes, d), dtype=np.float64)
        
    def fit(self, X):
        self.path_to[0,0] = 0
        self.extend_tree(node_id=0, X=X, depth=0)
        self.corrected_depth = np.array([
            (c_factor(k)+sum(path>-1))/c_factor(self.n)
            for i,(k,path) in enumerate(zip(self.node_size,self.path_to))
            if i<self.node_count
        ])

    def create_new_node(self, parent_id: int, direction:int) -> int:
        new_node_id = self.node_count
        self.node_count+=1
        self.path_to[new_node_id] = self.path_to[parent_id]
        self.path_to_Right_Left[new_node_id] = self.path_to_Right_Left[parent_id]
        self.path_to[new_node_id, self.depth[parent_id]+1] = new_node_id
        self.path_to_Right_Left[new_node_id, self.depth[parent_id]] = direction
        self.depth[new_node_id] = self.depth[parent_id]+1     
        return new_node_id
    
    def extend_tree(self, node_id: int, X: npt.NDArray, depth: int) -> None:
        stack = [(0, X, 0)] 
        
        while stack:
            node_id, data, depth = stack.pop()
            
            self.node_size[node_id] = len(data)
            if self.node_size[node_id] <= self.min_sample or depth >= self.max_depth:
                continue
            
            self.normals[node_id] = make_rand_vector(self.d - self.locked_dims, self.d)         
            
            dist = np.dot(np.ascontiguousarray(data), np.ascontiguousarray(self.normals[node_id]))
        
            if self.plus:
                self.intercepts[node_id] = np.random.normal(np.mean(dist),np.std(dist)*2)
            else:
                self.intercepts[node_id] = np.random.uniform(np.min(dist),np.max(dist))
            mask = dist <= self.intercepts[node_id]  
            
            X_left = data[mask]
            X_right = data[~mask,:]

            self.importances_left[node_id] = np.abs(self.normals[node_id])*(self.node_size[node_id]/(len(X_left)+1))
            self.importances_right[node_id] = np.abs(self.normals[node_id])*(self.node_size[node_id]/(len(X_right)+1))
            
            left_child = self.create_new_node(node_id,-1)
            right_child = self.create_new_node(node_id,1)
            
            self.child_left[node_id] = left_child
            self.child_right[node_id] = right_child

            stack.append((left_child, X_left, depth + 1))
            stack.append((right_child, X_right, depth + 1))

    def leaf_ids(self, X):
        return get_leaf_ids(X, self.child_left, self.child_right, self.normals, self.intercepts) 
         
                
    def apply(self, X):
        return self.path_to[self.leaf_ids(X)] 
    
    def predict(self,X,ids):
        return self.corrected_depth[ids],
    
    def importances(self,ids):
        importances,normals = calculate_importances(
            self.path_to[ids], 
            self.path_to_Right_Left[ids], 
            self.importances_left, 
            self.importances_right, 
            self.normals, 
            self.d
        )
        
        return importances, normals


class ExtendedIsolationForest():
    def __init__(self, plus, n_estimators=400, max_depth="auto", max_samples="auto"):
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.max_depth = max_depth
        self.plus=plus
        self.ids=None
        self.X=None
    
    @property
    def avg_number_of_nodes(self):
        return np.mean([T.node_count for T in self.trees])
        
    def fit(self, X, locked_dims=None):

        if locked_dims is None:
            locked_dims = 0

        if self.max_depth == "auto":
            self.max_depth = int(np.ceil(np.log2(self.max_samples)))
        subsample_size = np.min((self.max_samples, len(X)))
        self.trees = [ExtendedTree(subsample_size, X.shape[1], self.max_depth, locked_dims=locked_dims, plus=self.plus)
                      for _ in range(self.n_estimators)]
        for T in self.trees:
            T.fit(X[np.random.randint(len(X), size=subsample_size)])
            
    def compute_ids(self, X):
        if self.X is None or self.X.shape != X.shape:
            self.X = X
            self.ids = np.array([tree.leaf_ids(X) for tree in self.trees])

    def predict(self, X):
        self.compute_ids(X)
        predictions=[tree.predict(X,self.ids[i]) for i,tree in enumerate(self.trees)]
        values = np.array([p[0] for p in predictions])
        return np.power(2,-np.mean([value for value in values], axis=0))
    
    def _predict(self,X,p):
        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat

    def _importances(self, X, ids):
        importances = np.zeros(X.shape)
        normals = np.zeros(X.shape)
        for i,T in tqdm(enumerate(self.trees)):
            importance, normal = T.importances(ids[i])
            importances += importance
            normals += normal
        return importances/self.n_estimators, normals/self.n_estimators
    
    def global_importances(self, X,p=0.1):
        self.compute_ids(X)
        y_hat = self._predict(X,p)
        importances, normals = self._importances(X, self.ids)
        outliers_importances,outliers_normals = np.sum(importances[y_hat],axis=0),np.sum(normals[y_hat],axis=0)
        inliers_importances,inliers_normals = np.sum(importances[~y_hat],axis=0),np.sum(normals[~y_hat],axis=0)
        return (outliers_importances/outliers_normals)/(inliers_importances/inliers_normals)
    
    def local_importances(self, X, new_dataset = False):
        self.compute_ids(X)
        importances, normals = self._importances(X, self.ids)
        return importances/normals


class IsolationForest(ExtendedIsolationForest):
    def __init__(self,n_estimators=400, max_depth="auto", max_samples="auto"):
        super().__init__(plus=False,n_estimators=n_estimators,max_depth=max_depth,max_samples=max_samples)

    def fit(self, X):
        return super().fit(X, locked_dims=12)
    
    def decision_function_single_tree(self,tree_idx,X,p=0.1):
        self.compute_ids(X)
        pred=self.trees[tree_idx].predict(X,self.ids[tree_idx])[0]
        score=np.power(2,-pred)
        y_hat = np.array(score > sorted(score,reverse=True)[int(p*len(score))],dtype=int)
        return score,y_hat
    

    

"""
Use local_diffi,diffi_ib and get_iic functions to implement DIFFI inside this class: 

In local_diffi:
- iforest.estimators_ corresponds to self.trees
- iforest.estimator_samples_ coreesponds to self.ids 
- iforest.max_samples corresponds to self.max_samples (but be careful because in the case of wine max_samples stays 256 but it should be 129,
  so we should change its value aftert we do this subsample_size = np.min((self.max_samples, len(X)))
- Modify:

    X_outliers_ib = X_ib[np.where(as_ib < 0)]
    X_inliers_ib = X_ib[np.where(as_ib > 0)]

using the y_hat returned by decision_function_single_tree
"""