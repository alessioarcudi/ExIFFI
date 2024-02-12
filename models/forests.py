import numpy as np
from numba import njit


@njit
def c_norm(k):
    if k <=1: return 0
    h_k = np.log(k-1)+0.57721
    return 2*h_k-2*(k-1)/k


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
    

class ExtendedTree():
    def __init__(self,X,plus,locked_dims=None,distribution='normal_mean',eta=2):
        self.locked_dims=locked_dims
        self.plus=plus
        self.distribution=distribution
        self.eta=eta
        self.fit(X)
        
    def fit(self,X):
        self.psi = len(X)
        self.d = X.shape[1]
        self.max_depth = np.log2(self.psi)
        
        self.child_left = np.zeros(10*self.psi, dtype=int)
        self.child_right = np.zeros(10*self.psi, dtype=int)
        self.normals = np.zeros((10*self.psi,self.d), dtype=float)
        self.intercepts = np.zeros(10*self.psi, dtype=float)
        self.node_size = np.zeros(10*self.psi, dtype=int)
        self.depth = np.zeros(10*self.psi, dtype=int)
        self.path_to = np.zeros(10*self.psi,dtype=object)
        
        self.node_count = 1
        self.path_to[0] = [0]
        
        self.extend_tree(node_id=0, X=X, depth=0)
        self.corrected_depth = np.array([
            (c_norm(k)+len(path))/c_norm(self.psi)
            for i,(k,path) in enumerate(zip(self.node_size,self.path_to))
            if i<self.node_count
        ])
        
    def create_new_node(self, parent_id):
        new_node_id = self.node_count
        self.node_count+=1
        self.path_to[new_node_id] = self.path_to[parent_id]+[new_node_id]
        self.depth[new_node_id] = self.depth[parent_id]+1     
        return new_node_id
    
    def sample_normal_vector(self):
        # sample random d dimensinal vector 
        normals = np.random.randn(self.d)
        
        # align the hyperplane to one of the locked dimensions, fix norm to remove bias
        if self.locked_dims is not None:
            locked = normals[self.locked_dims]
            selected_one = np.argmax(locked) 
            excess_norm = np.sqrt(np.sum(np.square(locked)))
            locked = np.zeros_like(locked)
            locked[selected_one] = excess_norm
            normals[self.locked_dims] = locked
        
        return normals        
        
    def extend_tree(self,node_id, X, depth):
        self.node_size[node_id] = len(X)
        
        if depth >= self.max_depth or len(X)<=1:
            return
        
        # sample random normal vector
        self.normals[node_id] = self.sample_normal_vector()            
        
        # compute distances from plane (intercept in origin)
        dist = np.dot(X, self.normals[node_id])
        
        # sample intercept 
        if np.random.random() < self.plus:
            #sample intercept EIF+ with normal_mean
            if self.distribution == 'normal_mean':
                self.intercepts[node_id] = np.random.normal(np.mean(dist),np.std(dist)*self.eta)
            #sample intercept EIF+ with normal_median
            elif self.distribution == 'normal_median':
                self.intercepts[node_id] = np.random.normal(np.median(dist),np.std(dist)*self.eta)
            #sample intercept EIF+ with scaled_uniform
            elif self.distribution == 'scaled_uniform':
                self.intercepts[node_id] = np.random.uniform(np.min(dist)/self.eta,np.max(dist)*self.eta)
        else:
            #sample intercept EIF 
            self.intercepts[node_id] = np.random.uniform(np.min(dist),np.max(dist))

        
        
        # split X
        X_left = X[dist <= self.intercepts[node_id]]
        X_right = X[dist > self.intercepts[node_id]]        
        
        # add children 
        self.child_left[node_id] = self.create_new_node(node_id)
        self.child_right[node_id] = self.create_new_node(node_id)
        
        # recurse on children
        self.extend_tree(self.child_left[node_id], X_left, depth+1)        
        self.extend_tree(self.child_right[node_id], X_right, depth+1)
        
    def leaf_ids(self, X):
        return get_leaf_ids(X, self.child_left, self.child_right, self.normals, self.intercepts) 
                
    def apply(self, X):
        return self.path_to[self.leaf_ids(X)] 
    
    def predict(self, X):
        return self.corrected_depth[self.leaf_ids(X)]
    

class ExtendedIsolationForest():
    def __init__(self,
                 plus,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="auto",
                 distribution="normal_mean",
                 eta=2):
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.c_norm = c_norm(self.max_samples)
        self.plus=plus
        self.distribution=distribution
        self.eta=eta

        if contamination == "auto":
            self.contamination = 0.1
        else:
            if not (0 <= contamination <= 0.5):
                raise ValueError("Contamination must be between 0 and 0.5")
            self.contamination = contamination

    def set_distribution(
        self, distribution
    ):
        """
        Set the distribution to use in sampling the point p 
        for the cutting hyperplane 
        """
        if (distribution == 'normal_mean') or (distribution == 'normal_median') or (distribution == 'scaled_uniform'):
                self.distribution=distribution
        else:
                raise ValueError(f"Distribution {distribution} not supported")
        
    def set_eta(self,eta): 
        """
        Set the scaling factor eta to use in the distribution for the intercept point p in the cutting hyperplanes
        """
        self.eta=eta

    @property
    def contamination_(self):
        return self.contamination
    
    @property
    def distribution_(self):
        return self.distribution
    
    @property
    def eta_(self):
        return self.eta
        
    def fit(self, X, locked_dims=None):
        subsample_size = np.min((self.max_samples,len(X)))
        self.trees = [
            ExtendedTree(X[np.random.randint(len(X), size=subsample_size)],locked_dims=locked_dims,plus=self.plus,
                         distribution=self.distribution,eta=self.eta) 
            for _ in range(self.n_estimators)
        ]
        
    def Anomaly_Score(self, X):
        return np.power(2,-np.mean([tree.predict(X) for tree in self.trees], axis=0))
    
    def predict(self,X):
        An_score = self.Anomaly_Score(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(self.contamination*len(An_score))]
        return y_hat
    
    
class IsolationForest(ExtendedIsolationForest):       
    def fit(self, X):
        subsample_size = np.min((self.max_samples,len(X)))
        locked_dims = np.arange(X.shape[1],dtype=int)
        self.trees = [
            ExtendedTree(X[np.random.randint(len(X), size=subsample_size)],locked_dims=locked_dims) 
            for _ in range(self.n_estimators)
        ]
    
