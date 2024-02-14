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
import dask
from dask.distributed import Client

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
def calculate_importances(paths, directions, importances_left, importances_right, normals, d):
    # paths is now an encoded matrix representation of all paths
    # precomputed_contributions is a precomputed matrix of contributions for each nod
    left_paths = -np.ones_like(paths)
    left_mask = directions==-1
    left_paths[left_mask] = paths[left_mask]
    
    right_paths = -np.ones_like(paths)
    right_mask = directions==1
    right_paths[right_mask] = paths[right_mask]
    
    importances_sum = (importances_left[left_paths] + importances_right[right_paths]).sum(axis=1)
    normals_sum = np.abs(normals[paths]).sum(axis=1)
    return importances_sum, normals_sum

# @njit
# def calculate_importances(paths, directions, importances_left, importances_right, normals, d):
#     num_samples = paths.shape[0]
#     importances = np.zeros((num_samples, d))
#     normals_sum = np.zeros((num_samples, d))
    
#     for i in range(num_samples):
#         path = paths[i]
#         direction = directions[i]
#         for node_id in range(path.shape[0]):
#             node = path[node_id]
#             if direction[node_id] == 0:
#                 break
#             if direction[node_id] == -1:
#                 importances[i] += importances_left[node]
#             else:
#                 importances[i] += importances_right[node]
#             normals_sum[i] += np.abs(normals[node])
    
#     return importances, normals_sum


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
            (c_factor(k)+sum(path>0))/c_factor(self.n)
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
    
    def predict(self, X):
        return self.corrected_depth[self.leaf_ids(X)]
    
    def importances(self, X):
        importances,normals = calculate_importances(
            self.path_to[self.leaf_ids(X)], 
            self.path_to_Right_Left[self.leaf_ids(X)], 
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
        
    def fit(self, X, locked_dims=None):
        if not locked_dims:
            locked_dims = 0
        if self.max_depth == "auto":
            self.max_depth = int(2*np.ceil(np.log2(self.max_samples)))
        subsample_size = np.min((self.max_samples, len(X)))
        self.trees = [ExtendedTree(subsample_size, X.shape[1], 100, self.plus, locked_dims)
                      for _ in range(self.n_estimators)]
        for T in self.trees:
            T.fit(X[np.random.randint(len(X), size=subsample_size)])

    def predict(self, X):
        return np.power(2,-np.mean([tree.predict(X) for tree in self.trees], axis=0))
    
    def _predict(self,X,p):
        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat

    def _importances(self, X):
        importances = np.zeros(X.shape)
        normals = np.zeros(X.shape)
        for T in self.trees:
            importance, normal = T.importances(X)
            importances += importance
            normals += normal
        return importances/self.n_estimators, normals/self.n_estimators
    
    def global_importances(self, X,p=0.1):
        y_hat = self._predict(X,p)
        importances, normals = self._importances(X)
        outliers_importances,outliers_normals = np.mean(importances[y_hat],axis=0),np.mean(normals[y_hat],axis=0)
        inliers_importances,inliers_normals = np.mean(importances[~y_hat],axis=0),np.mean(normals[~y_hat],axis=0)
        return (outliers_importances/outliers_normals)/(inliers_importances/inliers_normals)
    
    def local_importances(self, X):
        importances, normals = self._importances(X)
        return importances/normals



    
