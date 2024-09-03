import numpy as np
from typing import Type
import time
from math import ceil 
import sys
sys.path.append("../models")
from model_reboot.sklearn_mod_functions import *
from sklearn.ensemble import IsolationForest


def diffi_ib(iforest, X, adjust_iic=True): # "ib" stands for "in-bag"
    """
    Compute the Global feature importances with the DIFFI model 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    iforest: Object
            Instance of the Isolation Forest model

    adjust_iic: bool
            Boolean variable used to decide weather to adjust the iic (Induced Imbalance Coefficient) scores in the final computation


    Returns
    ----------
    fi_ib:      np.array
            Feature Importance Scores
    exec_time:  float
            Execution Time 
        
    """
    # start time
    start = time.time()
    # initialization
    num_feat = X.shape[1] 
    estimators = iforest.estimators_
    cfi_outliers_ib = np.zeros(num_feat).astype('float') #this is the I_O
    cfi_inliers_ib = np.zeros(num_feat).astype('float') #this is the I_I
    counter_outliers_ib = np.zeros(num_feat).astype('int') #this is the C_O
    counter_inliers_ib = np.zeros(num_feat).astype('int') #this is the C_I
    in_bag_samples = iforest.estimators_samples_
    # for every iTree in the iForest
    for k, estimator in enumerate(estimators):
        # get in-bag samples indices
        in_bag_sample = list(in_bag_samples[k])
        # get in-bag samples (predicted inliers and predicted outliers)
        X_ib = X[in_bag_sample,:]
        as_ib = decision_function_single_tree(iforest, k, X_ib)
        X_outliers_ib = X_ib[np.where(as_ib < 0)]
        X_inliers_ib = X_ib[np.where(as_ib > 0)]
        if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0: 
            continue
        # compute relevant quantities
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        # compute node depths
        stack = [(0, -1)]  
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            # if we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        # OUTLIERS
        # compute IICs for outliers
        lambda_outliers_ib = _get_iic(estimator, X_outliers_ib, is_leaves, adjust_iic)
        # update cfi and counter for outliers
        node_indicator_all_points_outliers_ib = estimator.decision_path(X_outliers_ib)
        node_indicator_all_points_array_outliers_ib = node_indicator_all_points_outliers_ib.toarray()
        # for every point judged as abnormal
        for i in range(len(X_outliers_ib)):
            path = list(np.where(node_indicator_all_points_array_outliers_ib[i] == 1)[0])
            depth = node_depth[path[-1]]
            for node in path:
                current_feature = feature[node]
                if lambda_outliers_ib[node] == -1:
                    continue
                else:
                    cfi_outliers_ib[current_feature] += (1 / depth) * lambda_outliers_ib[node]
                    counter_outliers_ib[current_feature] += 1
        # INLIERS
        # compute IICs for inliers 
        lambda_inliers_ib = _get_iic(estimator, X_inliers_ib, is_leaves, adjust_iic)
        # update cfi and counter for inliers
        node_indicator_all_points_inliers_ib = estimator.decision_path(X_inliers_ib)
        node_indicator_all_points_array_inliers_ib = node_indicator_all_points_inliers_ib.toarray()
        # for every point judged as normal
        for i in range(len(X_inliers_ib)):
            path = list(np.where(node_indicator_all_points_array_inliers_ib[i] == 1)[0])
            depth = node_depth[path[-1]]
            for node in path:
                current_feature = feature[node]
                if lambda_inliers_ib[node] == -1:
                    continue
                else:
                    cfi_inliers_ib[current_feature] += (1 / depth) * lambda_inliers_ib[node]
                    counter_inliers_ib[current_feature] += 1
    # compute FI
    fi_outliers_ib = np.where(counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0)
    fi_inliers_ib = np.where(counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0)
    fi_ib = fi_outliers_ib / fi_inliers_ib
    end = time.time()
    exec_time = end - start
    return fi_ib, exec_time


def local_diffi(iforest, x):
    """
    Compute the Local feature importances with the DIFFI model 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    iforest: Object
            Instance of the Isolation Forest model

    x: np.array
            Input sample on which the Local Feature Importance must be computed


    Returns
    ----------
    fi:      np.array
            Local Feature Importance Scores
    exec_time:  float
            Execution Time 
        
    """
    # start time
    start = time.time()
    # initialization 
    estimators = iforest.estimators_
    cfi = np.zeros(len(x)).astype('float')
    counter = np.zeros(len(x)).astype('int')
    max_depth = int(np.ceil(np.log2(iforest.max_samples_)))
    # for every iTree in the iForest
    for estimator in estimators:
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        # compute node depths
        stack = [(0, -1)]  
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            # if test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        # update cumulative importance and counter
        x = x.reshape(1,-1)
        node_indicator = estimator.decision_path(x)
        node_indicator_array = node_indicator.toarray()
        path = list(np.where(node_indicator_array == 1)[1])
        leaf_depth = node_depth[path[-1]]
        for node in path:
            if not is_leaves[node]:
                current_feature = feature[node] 
                cfi[current_feature] += (1 / leaf_depth) - (1 / max_depth)
                counter[current_feature] += 1
    # compute FI
    fi = np.zeros(len(cfi))
    for i in range(len(cfi)):
        if counter[i] != 0:
            fi[i] = cfi[i] / counter[i]
    end = time.time()
    exec_time = end - start
    return fi, exec_time

def local_diffi_batch(X: np.array, model:Type[IsolationForest]=IsolationForest()):
    """Computes the Local Feature Importance scores for a set of input samples according to the DIFFI algorithm.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features) representing the input samples.

    Returns
    -------
    fi : numpy array of shape (n_samples, n_features) representing the Local Feature Importance scores.
    ord_idx : numpy array of shape (n_samples, n_features) representing the order of the features according to their Local Feature Importance scores.
    The samples are sorted in decreasing order of Feature Importance. 
    exec_time : float representing the execution time of the algorithm.
    """
    fi = []
    ord_idx = []
    exec_time = []
    for i in range(X.shape[0]):
        x_curr = X[i, :]
        fi_curr, exec_time_curr = model.local_diffi(x_curr)
        fi.append(fi_curr)
        ord_idx_curr = np.argsort(fi_curr)[::-1]
        ord_idx.append(ord_idx_curr)
        exec_time.append(exec_time_curr)
    fi = np.vstack(fi)
    ord_idx = np.vstack(ord_idx)
    return fi, ord_idx, exec_time

def _get_iic(estimator, predictions, is_leaves, adjust_iic):
    """
    Compute Induced Balance Coefficients (IIC) 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    estimator: Object
            Instance of the estimator, in most cases this would be an Isolation Forest model 

    predictions: np.array
            Estimator's predictions
    
    is_leaves:   List of bool
            For each node it says if it is a leaf or not 

     adjust_iic: bool
            Boolean variable used to decide weather to adjust the iic (Induced Imbalance Coefficient) scores in the final computation  

    Returns
    ----------
    fi_ib:      np.array
            Feature Importance Scores
    exec_time:  float
            Execution Time 
        
    """
    desired_min = 0.5
    desired_max = 1.0
    epsilon = 0.0
    n_nodes = estimator.tree_.node_count
    lambda_ = np.zeros(n_nodes)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    # compute samples in each node
    node_indicator_all_samples = estimator.decision_path(predictions).toarray() 
    num_samples_in_node = np.sum(node_indicator_all_samples, axis=0)
    # ASSIGN INDUCED IMBALANCE COEFFICIENTS (IIC)
    for node in range(n_nodes):
        # compute relevant quantities for current node
        num_samples_in_current_node = num_samples_in_node[node]
        num_samples_in_left_children = num_samples_in_node[children_left[node]]
        num_samples_in_right_children = num_samples_in_node[children_right[node]]        
        # if there is only 1 feasible split or node is leaf -> no IIC is assigned
        if num_samples_in_current_node == 0 or num_samples_in_current_node == 1 or is_leaves[node]:    
            lambda_[node] = -1         
        # if useless split -> assign epsilon
        elif num_samples_in_left_children == 0 or num_samples_in_right_children == 0:
            lambda_[node] = epsilon
        else:
            if num_samples_in_current_node%2==0:    # even
                current_min = 0.5
            else:   # odd
                current_min = ceil(num_samples_in_current_node/2)/num_samples_in_current_node
            current_max = (num_samples_in_current_node-1)/num_samples_in_current_node
            tmp = np.max([num_samples_in_left_children, num_samples_in_right_children]) / num_samples_in_current_node
            if adjust_iic and current_min!=current_max:
                lambda_[node] = ((tmp-current_min)/(current_max-current_min))*(desired_max-desired_min)+desired_min
            else:
                lambda_[node] = tmp
    return lambda_


