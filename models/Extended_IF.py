"""
Extended Isolation Forest model 
"""
import sys;sys.path.append('../src')
from src.utils import make_rand_vector,c_factor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

class ExtendedIF():
    """
    EIF/EIF_plus model implementation
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    n_trees: int
            Number of Isolation Trees composing the forest 
    max_depth: int
            Maximum depth at which a sample can be considered as isolated
    min_sample: int
            Minimum number of samples in a node where isolation is achieved 
    dims: int
            Number of degrees of freedom used in the separating hyperplanes. 
    subsample_size: int
            Subsample size used in each Isolation Tree
    forest: List
            List of objects from the class ExtendedTree
    plus: int
            This parameter is used to distinguish the EIF and the EIF_plus models. If plus=0 then the EIF model 
            will be used, if plus=1 than the EIF_plus model will be considered.      
    """
    def __init__(self, n_trees, max_depth = None, min_sample = None, dims = None, subsample_size = None, plus=1):
        self.n_trees                    = n_trees
        self.max_depth                  = max_depth
        self.min_sample                 = min_sample
        self.dims                       = dims
        self.subsample_size             = subsample_size
        self.forest                     = None
        self.plus                       = plus

    def fit(self,X):
        """
        Fit the EIF/EIF_plus model. 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        Returns
        ----------

        """
        if not self.dims:
            self.dims = X.shape[1]
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = np.inf

        self.forest = [ExtendedTree(self.dims, self.min_sample, self.max_depth,self.plus) for i in range(self.n_trees)]
        for x in self.forest:
            if not self.subsample_size:
                x.make_tree(X,0,0)
            else:
                X_sub = X[np.random.choice(X.shape[0], self.subsample_size, replace=False), :]
                x.make_tree(X_sub,0,0)

    def Anomaly_Score(self,X,algorithm=1):
        """
        Compute the Anomaly Score for an input dataset 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        algorithm: int
                This variable is used to decide weather to use the compute_paths or compute_paths2 function in the computation of the 
                Anomaly Scores.
        Returns
        ----------
        Returns the Anomaly Scores of all the samples contained in the input dataset X.            
        """
        mean_path = np.zeros(len(X))
        if algorithm == 1:
            for i in self.forest:
                mean_path+=i.compute_paths(X)

        elif algorithm == 0:
            for i in self.forest:
                mean_path+=i.compute_paths2(X,0)

        mean_path = mean_path/len(self.forest)
        c = c_factor(len(X))

        return 2**(-mean_path/c)

    def _predict(self,X,p):
        """
        Predict the anomalous or not anomalous nature of a set of input samples
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        p: int
                Contamination factor used to determine the threshold to apply to the Anomaly Score for the prediction
        Returns
        ----------
        y_hat: np.array
                Returns 0 for inliers and 1 for outliers          
        """
        An_score = self.Anomaly_Score(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat

    #??? 
    def evaluate(self,X,y,p):
        An_score = self.Anomaly_Score(X)
        m = np.c_[An_score,y]
        m = m[(-m[:,0]).argsort()]
        return np.sum(m[:int(p*len(X)),1])/int(p*len(X))

    def print_score_map(self,X,resolution,plot=None,features=[0,1]):
        '''
        Produce the Anomaly Score Scoremap.   
        --------------------------------------------------------------------------------
        
        Parameters
        ---------- 
        X: pd.DataFrame
                Input dataset
        resolution: int
                Scoremap resolution 
        plot: None
                Variable used to distinguish different ways of managing the plot settings. 
                By default the value of plot is set to None. 
        features: List
                List containing the pair of variables compared in the Scoremap.
                By default the value of features is [0,1]. 
        Returns
        ----------
        Returns the Anomaly Score Scoremap  
        '''
        if plot == None:
            fig,plot = plt.subplots(1,1,figsize=(10,10))
        mins = X[:,features].min(axis=0)
        maxs = X[:,features].max(axis=0)
        mins = list(mins-(maxs-mins)*3/10)
        maxs = list(maxs+(maxs-mins)*3/10)
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))

        means = np.mean(X,axis=0)
        feat_0 = xx.ravel()
        feat_1 = yy.ravel()
        dataset = np.array([x*np.ones(len(feat_0)) for x in means]).T
        dataset[:,features[0]] = feat_0
        dataset[:,features[1]] = feat_1        
        S1 = self.Anomaly_Score(dataset)
        S1 = S1.reshape(xx.shape)
        x= X.T[0]
        y= X.T[1]        

        levels = np.linspace(np.min(S1),np.max(S1),10)
        CS = plot.contourf(xx, yy, S1, levels, cmap=plt.cm.YlOrRd)
        cb = colorbar(CS, extend='max')
        cb.ax.set_yticks(fontsize = 12)
        cb.ax.set_ylabel("Anomaly Score",fontsize = 16)
        plot.scatter(x,y,s=15,c='None',edgecolor='k')
        plot.set_title("Anomaly Score scoremap with {} Degree of Freedom".format(self.dims),fontsize = 18)
        plot.set_xlabel("feature {}".format(features[0]),fontsize = 16)
        plot.set_ylabel("feature {}".format(features[1]),fontsize = 16)

            
    
    
class ExtendedTree():
    
    def __init__(self,dims,min_sample,max_depth,plus):
        ''' 
        Implementation of Isolation Trees for the EIF/EIF_plus models 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        dims: int
                Number of degrees of freedom used in the separating hyperplanes
        min_sample: int
                Minimum number of samples in a node where isolation is achieved 
        max_depth: int
                Maximum depth at which a sample can be considered as isolated
        plus: int
                This parameter is used to distinguish the EIF and the EIF_plus models. If plus=0 then the EIF model 
                will be used, if plus=1 than the EIF_plus model will be considered.
        '''
        self.dims                   = dims
        self.min_sample             = min_sample
        self.max_depth              = max_depth
        self.depth                  = 0
        self.right_son              = [0]
        self.left_son               = [0]
        self.nodes                  = {}
        self.plus                   = plus
    
    def make_tree(self,X,id,depth):
        ''' 
        Create an Isolation Tree using the separating hyperplanes 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        id: int 
                Node index 
        depth: int 
                Depth value 
        
        N.B The id and depth input parameters are needed because this is a recursive algorithm. At the first call id and depth 
        will be equal to 0. 

        '''
        if X.shape[0] <= self.min_sample:
            self.nodes[id] = {"point":None,"normal":None,"numerosity":len(X)}
        elif depth >= self.max_depth:
            self.nodes[id] = {"point":None,"normal":None,"numerosity":len(X)}
        else:
            n = make_rand_vector(self.dims,X.shape[1])
            
            val = X.dot(n)
            val_min = np.min(val)
            val_max = np.max(val)
            if np.random.random() < self.plus:
                s = np.random.normal(np.mean(val),np.std(val)*2)
            else:
                s = np.random.uniform(val_min,val_max)
            
            lefts = val>s
                
            self.nodes[id] = {"point":s,"normal":n,"numerosity":len(X)}
            
            idsx = len(self.nodes)
            self.left_son[id] = int(idsx)
            self.right_son.append(0)
            self.left_son.append(0)
            self.make_tree(X[lefts], idsx, depth+1)
            
            iddx = len(self.nodes)
            self.right_son.append(0)
            self.left_son.append(0)
            self.right_son[id] = int(iddx)
            self.make_tree(X[~lefts], iddx, depth+1)
    
    def compute_paths2(self, X, id, true_vec = None):
        ''' 
        Compute the path followed by sample from the root to the leaf where it is contained. The result of this 
        function is used for the Anomaly Score computation.  
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        id: int 
                Node index 
        true_vec: np.array
                The true_vec array has the same length as X. It is equal to 1 in correspondance of the nodes 
                where a sample passed, 0 otherwise.
                By default the value of true_vec is None
        Returns
        ----------
        Returns true_vec summed with two recursive calls, one on the left son and one on the right son.   
        ''' 
        if id==0:
            true_vec = np.ones(len(X))
        s = self.nodes[id]["point"]
        n = self.nodes[id]["normal"]

        if s is None:
            return true_vec*1
        else:
            val = np.array(X[true_vec==1].dot(n)-s>0)
            lefts = true_vec.copy()
            rights = true_vec.copy()
            lefts[true_vec==1] = val
            rights[true_vec==1] = np.logical_not(val)  
            return true_vec*1 + self.compute_paths2( X, int(self.left_son[id]), true_vec = lefts) + self.compute_paths2( X, int(self.right_son[id]), true_vec = rights)
 
    def compute_paths(self, X):
        ''' 
        Alternative method to compute the path followed by sample from the root to the leaf where it is contained. The result of this 
        function is used for the Anomaly Score computation.  
        This function is the iterative version of the compute_paths2 function. 
        --------------------------------------------------------------------------------
        
        Parameters
        ----------
        X: pd.DataFrame
                Input dataset   
        Returns
        ----------
        paths: List
                List of nodes encountered by a sample in its path towards the leaf in which it is contained.       
        ''' 
        paths = []
        for x in X:
            id=0
            k=1
            s = self.nodes[id]["point"]
            n = self.nodes[id]["normal"]
            while s is not None:
                val = x.dot(n)-s>0
                if val:
                    id = self.left_son[id]
                else:
                    id = self.right_son[id]
                s = self.nodes[id]["point"]
                n = self.nodes[id]["normal"]
                k+=1
            paths.append(k)
        return paths
    
    ''' 
    def predict(self,X, algorithm = 1):
        mean_path = np.zeros(len(X))
        if algorithm == 1:
            for i in self.forest:
                mean_path+=i.compute_paths(X)
                
        elif algorithm == 0:
            for i in self.forest:
                mean_path+=i.compute_paths2(X,0)

        mean_path = mean_path/len(self.forest)
        c = c_factor(len(X))
            
        return 2**(-mean_path/c)
    '''