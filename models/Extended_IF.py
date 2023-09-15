"""
Extended Isolation Forest model 
"""
from utils import make_rand_vector,c_factor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

class ExtendedIF():
    def __init__(self, n_trees, max_depth = None, min_sample = None, dims = None, subsample_size = None, plus=1):
        self.n_trees                    = n_trees
        self.max_depth                  = max_depth
        self.min_sample                 = min_sample
        self.dims                       = dims
        self.subsample_size             = subsample_size
        self.forest                     = None
        self.plus                       = plus

    def fit(self,X):
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

    def Anomaly_Score(self,X, algorithm = 1):
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

    def _predict(self,X,p):
        An_score = self.Anomaly_Score(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat

    def evaluate(self,X,y,p):
        An_score = self.Anomaly_Score(X)
        m = np.c_[An_score,y]
        m = m[(-m[:,0]).argsort()]
        return np.sum(m[:int(p*len(X)),1])/int(p*len(X))

    def print_score_map(self,X,risoluzione,plot=None, features = [0,1]):
        if plot == None:
            fig,plot = plt.subplots(1,1,figsize=(10,10))
        mins = X[:,features].min(axis=0)
        maxs = X[:,features].max(axis=0)
        mins = list(mins-(maxs-mins)*3/10)
        maxs = list(maxs+(maxs-mins)*3/10)
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], risoluzione), np.linspace(mins[1], maxs[1], risoluzione))

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
    
    def __init__(self,dims, min_sample, max_depth,plus):
        self.dims                   = dims
        self.min_sample             = min_sample
        self.max_depth              = max_depth
        self.depth                  = 0
        self.right_son              = [0]
        self.left_son               = [0]
        self.nodes                  = {}
        self.plus                   = plus
    
    def make_tree(self,X,id,depth):
        if X.shape[0] <= self.min_sample:
            self.nodes[id] = {"point":None,"normal":None,"numerosity":len(X)}
        elif depth >= self.max_depth:
            self.nodes[id] = {"point":None,"normal":None,"numerosity":len(X)}
        else:
            n = make_rand_vector(self.dims,X.shape[1])
            
            val = X.dot(n)
            val_min = np.min(val)
            val_max = np.max(val)
            # If self.plus=1 -> EIF+ Model 
            # If self.plus=0 -> EIF Model 
            if np.random.random() < self.plus:
                s = np.random.normal(np.mean(val), np.std(val)*2)
            else:
                s = np.random.uniform(val_min,np.max(val))
            
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