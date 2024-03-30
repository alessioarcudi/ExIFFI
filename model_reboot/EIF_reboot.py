from __future__ import annotations

from typing import ClassVar, Optional, List, Union
import numpy.typing as npt
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange, float64, int64, boolean
from numba.experimental import jitclass
from joblib import Parallel, delayed


from numba import njit
from numba.typed import List
import numpy as np
from tqdm import tqdm

@njit(cache=True)
def make_rand_vector(df:int,
                     dimensions:int) -> npt.NDArray[np.float64]:
    """
    Generate a random unitary vector in the unit ball with a maximum number of dimensions. 
    This vector will be successively used in the generation of the splitting hyperplanes.
    
    Args:
        df: Degrees of freedom
        dimensions: number of dimensions of the feature space

    Returns:
        vec: Random unitary vector in the unit ball
        
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



@njit(cache=True)
def c_factor(n: int) -> float:
    """
    Average path length of unsuccesful search in a binary search tree given n points.
    This is a constant factor that will be used as a normalization factor in the Anomaly Score calculation.
    
    Args:
        n: Number of data points for the BST.
        
    Returns:
        float: Average path length of unsuccesful search in a BST
        
    """
    if n <=1: return 0
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


@njit(cache=True)
def get_leaf_ids(X:np.array,
                 child_left:np.array,
                 child_right:np.array,
                 normals:np.array,
                 intercepts:np.array) -> np.array:
        
        """
        Get the leaf node ids for each data point in the dataset.

        Args:
            X: Data points
            child_left: Left child node ids
            child_right: Right child node ids
            normals: Normal vectors of the splitting hyperplanes
            intercepts: Intercept values of the splitting hyperplanes

        Returns:
            np.array: Leaf node ids for each data point in the dataset.
        """
        res = []
        for x in X:
            node_id = 0
            while child_left[node_id] or child_right[node_id]:
                d = np.dot(np.ascontiguousarray(x),np.ascontiguousarray(normals[node_id]))
                node_id = child_left[node_id] if d <= intercepts[node_id] else child_right[node_id]        
            res.append(int(node_id))
        return np.array(res)

@njit(cache=True)
def calculate_importances(paths:np.ndarray,
                          directions:np.ndarray, 
                          importances_left:np.ndarray, 
                          importances_right:np.ndarray, 
                          normals:np.ndarray,
                          d:int):
    
    """
    Calculate the importances of the features for the given paths and directions.

    Args:
        paths: Paths to the leaf nodes
        directions: Directions to the leaf nodes
        importances_left: Importances of the left child nodes
        importances_right: Importances of the right child nodes
        normals: Normal vectors of the splitting hyperplanes
        d: Number of dimensions in the dataset

    Returns:
        np.array: Importances of the features for the given paths and directions.
    """

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
    ("max_nodes", int64),
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
    ("importances_left", float64[:, :]),
    ("eta", float64)
]

@jitclass(tree_spec)
class ExtendedTree:

    """
    Class that represents an Isolation Tree in the Extended Isolation Forest model.

     
    Attributes:
        plus: Boolean flag to indicate if the model is a `EIF` or `EIF+`. Defaults to True (i.e. `EIF+`)
        locked_dims: Number of dimensions to be locked in the model. Defaults to 0
        max_depth: Maximum depth of the tree
        min_sample: Minimum number of samples in a node. Defaults to 1
        n: Number of samples in the dataset
        d: Number of dimensions in the dataset
        node_count: Counter for the number of nodes in the tree
        max_nodes: Maximum number of nodes in the tree. Defaults to 10000
        path_to: Array to store the path to the leaf nodes
        path_to_Right_Left: Array to store the path to the leaf nodes with directions
        child_left: Array to store the left child nodes
        child_right: Array to store the right child nodes
        normals: Array to store the normal vectors of the splitting hyperplanes
        intercepts: Array to store the intercept values of the splitting hyperplanes
        node_size: Array to store the size of the nodes
        depth: Array to store the depth of the nodes
        corrected_depth: Array to store the corrected depth of the nodes
        importances_right: Array to store the importances of the right child nodes
        importances_left: Array to store the importances of the left child nodes
        eta: Eta value for the model. Defaults to 1.5
    

    """

    def __init__(self,
                 n:int,
                 d:int,
                 max_depth:int,
                 locked_dims:int=0,
                 min_sample:int=1,
                 plus:bool=True,
                 max_nodes:int=10000,
                 eta:float=1.5):

        self.plus = plus
        self.locked_dims = locked_dims 
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.n = n
        self.d = d
        self.node_count = 1
        self.max_nodes = max_nodes
        self.eta = eta

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
        
    def fit(self, X:np.array) -> None:

        """
        Fit the model to the dataset.

        Args:
            X: Input dataset

        Returns:
            None: The method fits the model and does not return any value.
        """

        self.path_to[0,0] = 0
        self.extend_tree(node_id=0, X=X, depth=0)
        self.corrected_depth = np.array([
            (c_factor(k)+sum(path>-1))/c_factor(self.n)
            for i,(k,path) in enumerate(zip(self.node_size,self.path_to))
            if i<self.node_count
        ])

    def create_new_node(self,
                        parent_id:int,
                        direction:int) -> int:

        """
        Create a new node in the tree.

        Args:
            parent_id: Parent node id
            direction: Direction to the new node

        Returns:
            int: New node id

        """

        new_node_id = self.node_count
        self.node_count+=1
        self.path_to[new_node_id] = self.path_to[parent_id]
        self.path_to_Right_Left[new_node_id] = self.path_to_Right_Left[parent_id]
        self.path_to[new_node_id, self.depth[parent_id]+1] = new_node_id
        self.path_to_Right_Left[new_node_id, self.depth[parent_id]] = direction
        self.depth[new_node_id] = self.depth[parent_id]+1     
        return new_node_id
    
    def extend_tree(self,
                    node_id:int,
                    X:npt.NDArray,
                    depth: int) -> None:
        
        """
        Extend the tree to the given node.

        Args:
            node_id: Node id
            X: Input dataset
            depth: Depth of the node
        
        Returns:
            None: The method extends the tree and does not return any value.
        """

        stack = [(0, X, 0)] 
        
        while stack:
            node_id, data, depth = stack.pop()
            
            self.node_size[node_id] = len(data)
            if self.node_size[node_id] <= self.min_sample or depth >= self.max_depth:
                continue
            
            self.normals[node_id] = make_rand_vector(self.d - self.locked_dims, self.d)         
            
            dist = np.dot(np.ascontiguousarray(data), np.ascontiguousarray(self.normals[node_id]))
        
            if self.plus:
                #self.intercepts[node_id] = np.random.normal(np.mean(dist),np.std(dist)*self.eta)
                self.intercepts[node_id] = np.random.normal(np.mean(dist),np.std(dist)*self.eta)
            else:
                self.intercepts[node_id] = np.random.uniform(np.min(dist),np.max(dist))
            mask = dist <= self.intercepts[node_id]  
            
            X_left = data[mask]
            X_right = data[~mask,:]

            self.importances_left[node_id] = np.abs(self.normals[node_id])*(self.node_size[node_id]/(len(X_left)+1))
            self.importances_right[node_id] = np.abs(self.normals[node_id])*self.node_size[node_id]/(len(X_right)+1)
            
            left_child = self.create_new_node(node_id,-1)
            right_child = self.create_new_node(node_id,1)
            
            self.child_left[node_id] = left_child
            self.child_right[node_id] = right_child

            stack.append((left_child, X_left, depth + 1))
            stack.append((right_child, X_right, depth + 1))
            
            if self.node_count >= self.max_nodes:
                raise ValueError("Max number of nodes reached")

    def leaf_ids(self, X) -> np.array:
        """
        Get the leaf node ids for each data point in the dataset.

        This is a stub method of `get_leaf_ids`.

        Args:
            X: Input dataset

        Returns:
            np.array: Leaf node ids for each data point in the dataset.
        """
        return get_leaf_ids(X, self.child_left, self.child_right, self.normals, self.intercepts) 
         
                
    def apply(self, X):
        """
        Update the `path_to` attribute with the path to the leaf nodes for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            None: The method updates `path_to` and does not return any value.
        """
        return self.path_to[self.leaf_ids(X)] 
    
    def predict(self,
                ids:np.array) -> np.array:

        """
        Predict the anomaly score for each data point in the dataset.

        Args:
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            np.array: Anomaly score for each data point in the dataset.
        """
        return self.corrected_depth[ids],
    
    def importances(self,ids:np.array) -> tuple[np.array,np.array]:

        """
        Compute the importances of the features for the given leaf node ids.

        Args:
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            tuple[np.array,np.array]: Importances of the features for the given leaf node ids and the normal vectors.
        """
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

    """
    Class that represents the Extended Isolation Forest model.

    Attributes:
        n_estimators: Number of trees in the model. Defaults to 400
        max_samples: Maximum number of samples in a node. Defaults to 256
        max_depth: Maximum depth of the trees. Defaults to "auto"
        plus: Boolean flag to indicate if the model is a `EIF` or `EIF+`.
        name: Name of the model
        ids: Leaf node ids for each data point in the dataset. Defaults to None
        X: Input dataset. Defaults to None
        eta: Eta value for the model. Defaults to 1.5
        avg_number_of_nodes: Average number of nodes in the trees
    
    """

    def __init__(self,
                 plus:bool,
                 n_estimators:int=400,
                 max_depth:Union[str,int]="auto",
                 max_samples:Union[str,int]="auto",
                 eta:float = 1.5):
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.max_depth = max_depth
        self.plus=plus
        self.name="EIF"+"+"*int(plus)
        self.ids=None
        self.X=None
        self.eta=eta
    
    @property
    def avg_number_of_nodes(self):
        return np.mean([T.node_count for T in self.trees])
        
    def fit(self, X:np.array, locked_dims=None) -> None:

        """
        Fit the model to the dataset.

        Args:
            X: Input dataset
            locked_dims: Number of dimensions to be locked in the model. Defaults to None

        Returns:
            None: The method fits the model and does not return any value.
        """

        self.ids = None
        if not locked_dims:
            locked_dims = 0

        if self.max_depth == "auto":
            self.max_depth = int(np.ceil(np.log2(self.max_samples)))
        subsample_size = np.min((self.max_samples, len(X)))
        self.trees = [ExtendedTree(subsample_size, X.shape[1], self.max_depth, locked_dims=locked_dims, plus=self.plus, eta=self.eta)
                      for _ in range(self.n_estimators)]
        for T in self.trees:
            T.fit(X[np.random.randint(len(X), size=subsample_size)])
            
    def compute_ids(self, X:np.array) -> None:
        
        """
        Compute the leaf node ids for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            None: The method computes the leaf node ids and does not return any value.
        """
        if self.ids is None or self.X.shape != X.shape:
            self.X = X
            self.ids = np.array([tree.leaf_ids(X) for tree in self.trees])

    def predict(self, X:np.array) -> np.array:

        """
        Predict the anomaly score for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            np.array: Anomaly score for each data point in the dataset.
        """
        self.compute_ids(X)
        predictions=[tree.predict(X,self.ids[i]) for i,tree in enumerate(self.trees)]
        values = np.array([p[0] for p in predictions])
        return np.power(2,-np.mean([value for value in values], axis=0))
    
    def _predict(self,
                 X:np.array,
                 p:float) -> np.array:
        """
        Predict the class of each data point (i.e. inlier or outlier) based on the anomaly score.

        Args:
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score)

        Returns:
            np.array: Predicted class for each data point in the dataset.
        """
        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat

    def _importances(self,
                     X:np.array,
                     ids:np.array) -> tuple[np.array,np.array]:

        """
        Compute the importances of the features for the given leaf node ids.

        Args:
            X: Input dataset
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            tuple[np.array,np.array]: Importances of the features for the given leaf node ids and the normal vectors.

        """
        importances = np.zeros(X.shape)
        normals = np.zeros(X.shape)
        for i,T in enumerate(self.trees):
            importance, normal = T.importances(ids[i])
            importances += importance
            normals += normal
        return importances/self.n_estimators, normals/self.n_estimators
    
    def global_importances(self,
                           X:np.array,
                           p:float=0.1) -> np.array:

        """
        Compute the global importances of the features for the dataset.

        Args:
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score). Defaults to 0.1

        Returns:
            np.array: Global importances of the features for the dataset.
        """

        self.compute_ids(X)
        y_hat = self._predict(X,p)
        importances, normals = self._importances(X, self.ids)
        outliers_importances,outliers_normals = np.sum(importances[y_hat],axis=0),np.sum(normals[y_hat],axis=0) 
        inliers_importances,inliers_normals = np.sum(importances[~y_hat],axis=0),np.sum(normals[~y_hat],axis=0)
        return (outliers_importances/outliers_normals)/(inliers_importances/inliers_normals)
    
    def local_importances(self,
                          X:np.array) -> np.array:

        """
        Compute the local importances of the features for the dataset.

        Args:
            X: Input dataset

        Returns:
            np.array: Local importances of the features for the dataset.
        """
        
        self.compute_ids(X)
        importances, normals = self._importances(X, self.ids)
        return importances/normals


class IsolationForest(ExtendedIsolationForest):

    """
    Class that represents the Isolation Forest model. 
    
    This is a subclass of `ExtendedIsolationForest` with the `plus` attribute set to False and the 
    `locked_dims` attribute set to the number of dimensions minus one.

    """

    def __init__(self,
                 n_estimators:int=400,
                 max_depth:Union[str,int]="auto",
                 max_samples:Union[str,int]="auto") -> None:

        """
        Initialize the Isolation Forest model using the `__init__` method of the `ExtendedIsolationForest` class.

        Args:

            n_estimators: Number of trees in the model. Defaults to 400
            max_depth: Maximum depth of the trees. Defaults to "auto"
            max_samples: Maximum number of samples in a node. Defaults to "auto"

        Returns:
            None: The method initializes the model and does not return any value.

        """
        super().__init__(plus=False,n_estimators=n_estimators,max_depth=max_depth,max_samples=max_samples)
        self.name="IF"
            
    def fit(self,
            X:np.array) -> None:
        
        """
        Fit the model to the dataset.

        Args:
            X: Input dataset

        Returns:
            None: The method fits the model and does not return any value.
        """

        return super().fit(X, locked_dims=X.shape[1]-1)
    
    def decision_function_single_tree(self,
                                      tree_idx:int,
                                      X:np.array,
                                      p:float=0.1) -> tuple[np.array,np.array]:
        
        """
        Predict the anomaly score for each data point in the dataset using a single tree.

        Args:
            tree_idx: Index of the tree
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score). Defaults to 0.1

        Returns:
            tuple[np.array,np.array]: Anomaly score for each data point in the dataset and the predicted class for each data point in the dataset.
        """

        self.compute_ids(X)
        pred=self.trees[tree_idx].predict(X,self.ids[tree_idx])[0]
        score=np.power(2,-pred)
        y_hat = np.array(score > sorted(score,reverse=True)[int(p*len(score))],dtype=int)
        return score,y_hat
    

