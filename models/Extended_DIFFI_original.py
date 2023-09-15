import sys;sys.path.append("./models")
from models.Extended_IF import ExtendedIF,ExtendedTree
import numpy as np


class Extended_DIFFI_tree(ExtendedTree):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.importances = []
        self.sum_normals = []
    
    def make_importance(self,X,depth_based):
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

    
    def Importances(self, X, calculate, sovrascrivi,depth_based):
        if (self.sum_importances_matrix is None) or calculate:     
            sum_importances_matrix = np.zeros_like(X,dtype='float64')
            sum_normal_vectors_matrix = np.zeros_like(X,dtype='float64')
            k=0
            for i in self.forest:
                importances_matrix, normal_vectors_matrix = i.make_importance(X,depth_based)
                sum_importances_matrix += importances_matrix/self.n_trees
                sum_normal_vectors_matrix += normal_vectors_matrix/self.n_trees
                k+=1
            if sovrascrivi:
                self.sum_importances_matrix = sum_importances_matrix/self.n_trees
                self.sum_normal_vectors_matrix = sum_normal_vectors_matrix/self.n_trees
            return sum_importances_matrix,sum_normal_vectors_matrix
        else:
            return self.sum_importances_matrix,self.sum_normal_vectors_matrix

    
    def Global_importance(self, X, calculate, sovrascrivi, depth_based=False):
        anomaly_scores = self.Anomaly_Score(X)
        ind = np.argpartition(anomaly_scores,-int(0.1*len(X)))[-int(0.1*len(X)):]
        
        importances_matrix, normal_vectors_matrix = self.Importances(X, calculate, sovrascrivi, depth_based)
        
        Outliers_mean_importance_vector = np.mean(importances_matrix[ind],axis = 0)
        Inliers_mean_Importance_vector = np.mean(importances_matrix[np.delete(range(len(importances_matrix)),ind)], axis = 0)
        
        Outliers_mean_normal_vector = np.mean(normal_vectors_matrix[ind],axis = 0)
        Inliers_mean_normal_vector = np.mean(normal_vectors_matrix[np.delete(range(len(importances_matrix)),ind)], axis = 0)
        
        return (Outliers_mean_importance_vector/Outliers_mean_normal_vector) / (Inliers_mean_Importance_vector/Inliers_mean_normal_vector) - 1
    
    def Local_importances(self, X, calculate, sovrascrivi,depth_based=False):
        importances_matrix, normal_vectors_matrix = self.Importances(X, calculate, sovrascrivi,depth_based)
        return importances_matrix/normal_vectors_matrix
    