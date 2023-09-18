import sys;
sys.path.append("./models")
from models.Extended_IF import ExtendedIF,ExtendedTree
import numpy as np


class Extended_DIFFI_tree(ExtendedTree):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.importances = []
        self.sum_normals = []
    
    def make_importance(self,X,depth_based):
        ''' 
        Compute the Importance Scores for each node along the Isolation Trees.  
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function. 
        
        Returns
        ----------
        Importances_list: np.array
                List with the Importances values for all the nodes in the Isolation Tree. 
        Normal_vectors_list: np.array
                List of all the normal vectors used in the splitting hyperplane creation.  
        ''' 
        Importances_list = []
        Normal_vectors_list = []
        for x in X:
            importance = np.zeros(len(x))
            sum_normal = np.zeros(len(x))
            id = 0
            s = self.nodes[id]["point"]
            n = self.nodes[id]["normal"]
            N = self.nodes[id]["numerosity"]
            depth = 0
            while s is not None:
                
                val = x.dot(n)-s>0
                old_id = id
                if val:
                    id = self.left_son[id]
                    sum_normal += np.abs(n)
                    if depth_based == True:
                        singular_importance = np.abs(n)*(N/(self.nodes[id]["numerosity"]+1)) *1/(1+depth)
                        importance += singular_importance
                        self.nodes[old_id].setdefault("left_importance_depth", singular_importance)
                        self.nodes[old_id].setdefault("depth", depth)
                    else:
                        singular_importance = np.abs(n)*(N/(self.nodes[id]["numerosity"]+1)) 
                        importance += singular_importance
                        self.nodes[old_id].setdefault("left_importance", singular_importance)
                        self.nodes[old_id].setdefault("depth", depth)
                    depth+=1
                else:
                    id = self.right_son[id]
                    sum_normal += np.abs(n)
                    if depth_based == True:
                        singular_importance =np.abs(n)*(N/(self.nodes[id]["numerosity"]+1)) *1/(1+depth)
                        importance += singular_importance
                        self.nodes[old_id].setdefault("right_importance_depth", singular_importance)
                        self.nodes[old_id].setdefault("depth", depth)
                    else:
                        singular_importance = np.abs(n)*(N/(self.nodes[id]["numerosity"]+1))
                        importance += singular_importance 
                        self.nodes[old_id].setdefault("right_importance", singular_importance)
                        self.nodes[old_id].setdefault("depth", depth)
                    depth+=1
                s = self.nodes[id]["point"]
                n = self.nodes[id]["normal"]
                N = self.nodes[id]["numerosity"]
            Importances_list.append(importance)
            Normal_vectors_list.append(sum_normal)
        return np.array(Importances_list),np.array(Normal_vectors_list)
        

class Extended_DIFFI_original(ExtendedIF):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.sum_importances_matrix = None
        self.sum_normal_vectors_matrix = None
        self.plus=kwarg.get('plus')
            
    def fit(self,X):
        """
        Fit the ExIFFI model. 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset

        """
        if not self.dims:
            self.dims = X.shape[1]
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = np.inf
            
        self.forest = [Extended_DIFFI_tree(self.dims, self.min_sample, self.max_depth,self.plus) for i in range(self.n_trees)]
        self.subsets = []
        for x in self.forest:
            if not self.subsample_size or self.subsample_size>X.shape[0]:
                x.make_tree(X,0,0)
            else:
                indx = np.random.choice(X.shape[0], self.subsample_size, replace=False)
                X_sub = X[indx, :]
                x.make_tree(X_sub,0,0)
                self.subsets.append(indx)

    
    def Importances(self,X,calculate,overwrite,depth_based):
        ''' 
        Obtain the sum of the Importance scores computed along all the Isolation Trees, with the make_importance
        function.   
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function. 
        calculate: bool
                If calculate is True the Importances Sum Matrix and the Normal Vectors Sum Matrix are initialized to 0 
        overwrite: bool
                Boolean variable used to decide weather to overwrite evrytime the value inserted in sum_importances_matrix and 
                in sum_normal_vectors_matrix. 
        
        Returns
        ----------
        sum_importances_matrix: np.array
                2-dimensional array containing,for each sample, the sum of the importance scores obtained by the nodes in which 
                it was included.
        sum_normal_vectors_matrix: np.array
                2-dimensional array containing,for each sample, the sum of the normal vectors used to create the 
                splitting hyperplances of the nodes in which it was included.

        ''' 
        if (self.sum_importances_matrix is None) or calculate:     
            sum_importances_matrix = np.zeros_like(X,dtype='float64')
            sum_normal_vectors_matrix = np.zeros_like(X,dtype='float64')
            k=0
            for i in self.forest:
                importances_matrix, normal_vectors_matrix = i.make_importance(X,depth_based)
                sum_importances_matrix += importances_matrix/self.n_trees
                sum_normal_vectors_matrix += normal_vectors_matrix/self.n_trees
                k+=1
            if overwrite:
                self.sum_importances_matrix = sum_importances_matrix/self.n_trees
                self.sum_normal_vectors_matrix = sum_normal_vectors_matrix/self.n_trees
            return sum_importances_matrix,sum_normal_vectors_matrix
        else:
            return self.sum_importances_matrix,self.sum_normal_vectors_matrix

    
    def Global_importance(self, X, calculate, overwrite, depth_based=False):
        ''' 
        Compute the Global Feature Importance vector for a set of input samples 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        calculate: bool
                Used to call the Importances function 
        overwrite: bool
                Used to call the Importances function. 
         depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function. 
                By default the value of depth_based is False.
        
        Returns
        ----------
        Global_Importance: np.array 
        Array containig a Global Feature Importance Score for each feature in the dataset. 

        ''' 
        anomaly_scores = self.Anomaly_Score(X)
        ind = np.argpartition(anomaly_scores,-int(0.1*len(X)))[-int(0.1*len(X)):]
        
        importances_matrix, normal_vectors_matrix = self.Importances(X, calculate, overwrite, depth_based)
        
        Outliers_mean_importance_vector = np.mean(importances_matrix[ind],axis = 0)
        Inliers_mean_Importance_vector = np.mean(importances_matrix[np.delete(range(len(importances_matrix)),ind)], axis = 0)
        
        Outliers_mean_normal_vector = np.mean(normal_vectors_matrix[ind],axis = 0)
        Inliers_mean_normal_vector = np.mean(normal_vectors_matrix[np.delete(range(len(importances_matrix)),ind)], axis = 0)
        
        return (Outliers_mean_importance_vector/Outliers_mean_normal_vector) / (Inliers_mean_Importance_vector/Inliers_mean_normal_vector) - 1
    
    def Local_importances(self, X, calculate, overwrite,depth_based=False):
        ''' 
        Compute the Local Feature Importance vector for a set of input samples 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        calculate: bool
                Used to call the Importances function 
        overwrite: bool
                Used to call the Importances function. 
         depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function. 
                By default the value of depth_based is False.
        
        Returns
        ----------
        Local_Importance: np.array 
        Array containig a Local Feature Importance Score for each feature in the dataset. 

        ''' 
        importances_matrix, normal_vectors_matrix = self.Importances(X, calculate, overwrite,depth_based)
        return importances_matrix/normal_vectors_matrix
    