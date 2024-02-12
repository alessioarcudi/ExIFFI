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
                d = np.dot(x,normals[node_id])
                node_id = child_left[node_id] if d <= intercepts[node_id] else child_right[node_id]        
            res.append(int(node_id))
        return np.array(res)

tree_spec = [
    ("plus", boolean),
    ("locked_dims", int64),
    ("max_depth", int64),
    ("min_sample", int64),
    ("n", int64),
    ("d", int64),
    ("node_count", int64),
    ("path_to", int64[:, :]),     
    ("child_left", int64[:]),
    ("child_right", int64[:]),
    ("normals", float64[:, :]),
    ("intercepts", float64[:]),
    ("node_size", int64[:]),      
    ("depth", int64[:]),          
    ("corrected_depth", float64[:])
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

        self.path_to = np.zeros((max_nodes, max_depth+1), dtype=np.int64)
        self.child_left = np.zeros(max_nodes, dtype=np.int64)
        self.child_right = np.zeros(max_nodes, dtype=np.int64)
        self.normals = np.zeros((max_nodes, d), dtype=np.float64)
        self.intercepts = np.zeros(max_nodes, dtype=np.float64)
        self.node_size = np.zeros(max_nodes, dtype=np.int64)
        self.depth = np.zeros(max_nodes, dtype=np.int64)
        self.corrected_depth = np.zeros(max_nodes, dtype=np.float64)

    def fit(self, X):
        self.path_to[0,0] = 0
        
        self.extend_tree(node_id=0, X=X, depth=0)
        self.corrected_depth = np.array([
            (c_factor(k)+sum(path>0))/c_factor(self.n)
            for i,(k,path) in enumerate(zip(self.node_size,self.path_to))
            if i<self.node_count
        ])

    def create_new_node(self, parent_id: int) -> int:
        new_node_id = self.node_count
        self.node_count+=1
        self.path_to[new_node_id] = self.path_to[parent_id]
        self.path_to[new_node_id, self.depth[parent_id]+1] = new_node_id
        self.depth[new_node_id] = self.depth[parent_id]+1     
        return new_node_id
    
    def extend_tree(self, node_id: int, X: npt.NDArray, depth: int) -> None:
        stack = [(0, X, 0)] 
        
        while stack:
            node_id, data, depth = stack.pop()
            
            if len(data) <= self.min_sample or depth >= self.max_depth:
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
            
            left_child = self.create_new_node(node_id)
            right_child = self.create_new_node(node_id)
            
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


class ExtendedIsolationForest():
    def __init__(self, plus, n_estimators=400, max_samples="auto"):
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.plus=plus
        
    def fit(self, X, locked_dims=None):
        if not locked_dims:
            locked_dims = 0
        subsample_size = np.min((self.max_samples, len(X)))
        self.trees = [ExtendedTree(subsample_size, X.shape[1], int(np.log2(X.shape[0]))+1, self.plus, locked_dims)
                      for _ in range(self.n_estimators)]
        for T in self.trees:
            T.fit(X[np.random.randint(len(X), size=subsample_size)])

        
    def predict(self, X):
        return np.power(2,-np.mean([tree.predict(X) for tree in self.trees], axis=0))
    
    def _predict(self,X,p):
        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat
        


    
