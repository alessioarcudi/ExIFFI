import numpy as np
import pandas as pd
import pickle
from scipy.stats import kendalltau
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
#from ACME import ACME
import sklearn

from models.Extended_DIFFI_original import *

from DIFFI_master.interpretability_module import diffi_ib

#from utils.utils import *

#functions used for the compare feature task
    
    
def make_confidence_interval_file(Importances, imp_alg, name):
    K = np.array([x.values.T[0] for x in Importances[imp_alg]["importances"]])
    M = mean_confidence_interval_importances(np.array(K).T)
    path = '../results/compare_features/results/conf_interval/'+imp_alg+"_"+name+".pkl"
    with open(path, 'wb') as f:
        pickle.dump(M, f)
        
def create_Importances_dict(n_trees, precision, subsample_size, X_train, X_test, y, name):
    '''
    Importances = { "ExIFFI":{"importances":[],"robustness":[]},
                    "ExDIFFI":{"importances":[],"robustness":[]},
                    "DIFFI":{"importances":[],"robustness":[]}
                }
    '''

    Importances = { "ExIFFI":{"importances":[]},
                   "ExIFFI_plus":{"importances":[]},
                    "DIFFI":{"importances":[]}
                }

    score = {"IF":[],
            "EIF":[],
            'EIF_plus':[]
            }

    for i in tqdm(range(precision)):
        #fit model and save precision of EIF
        EDIFFI = Extended_DIFFI_original(n_trees, max_depth = 100, subsample_size = subsample_size,plus=0)
        EDIFFI.fit(X_train)
        scoreeif = EDIFFI.predict(X_test)
        avg_prec = sklearn.metrics.average_precision_score(y,scoreeif)

        #fit model and save precision of EIF+
        EDIFFI_plus = Extended_DIFFI_original(n_trees, max_depth = 100, subsample_size = subsample_size,plus=1)
        EDIFFI_plus.fit(X_train)
        scoreeif_plus = EDIFFI_plus.predict(X_test)
        avg_prec_plus = sklearn.metrics.average_precision_score(y,scoreeif_plus)

        score["EIF"].append(avg_prec)
        score["EIF_plus"].append(avg_prec_plus)
        
        #fit model and save precision of IF
        if subsample_size is None:
            iforest = IsolationForest().fit(X_train)
        else:
            iforest = IsolationForest(max_samples=subsample_size).fit(X_train)
        #Here we take the opposite of score_samples and add 0.5 to it because score_samples 
        #returns the opposito the anomaly score defined in the original IF paper so we want
        #to get back to the original score definition.
        scoreif = -1*iforest.score_samples(X_test)+0.5
        precision_IF = sklearn.metrics.average_precision_score(y,scoreif)
        score["IF"].append(precision_IF)
        
        
        
        importances_ediffi = EDIFFI.Global_importance(X_test,True,False,depth_based = False)
        importances_ediffi_plus = EDIFFI_plus.Global_importance(X_test,True,False,depth_based = False)
        #importances_ediffi_depth = EDIFFI.Global_importance(X_test,True,False,depth_based = True)
        
        fi_ib, exec_time = diffi_ib(iforest, X_test, adjust_iic=True)
        importances_diffi=pd.DataFrame(fi_ib)
        
        Importances["ExIFFI"]["importances"].append(pd.DataFrame(importances_ediffi))
        #Importances["ExIFFI"]["robustness"].append(extract_order(pd.DataFrame(importances_ediffi)))
        Importances["ExIFFI_plus"]["importances"].append(pd.DataFrame(importances_ediffi_plus))
        #Importances["ExDIFFI"]["importances"].append(pd.DataFrame(importances_ediffi_depth))
        #Importances["ExDIFFI"]["robustness"].append(extract_order(pd.DataFrame(importances_ediffi_depth)))
        Importances["DIFFI"]["importances"].append(importances_diffi)
        #Importances["DIFFI"]["robustness"].append(extract_order(importances_diffi))
        
    path = 'c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\results\\compare_features\\results\\Importances_dict_davide\\'+name+".pkl"
    with open(path, 'wb') as f:
        pickle.dump(Importances, f)
    return Importances,score

def make_importances_file(Importances,name):
    importances_e_diffi=pd.DataFrame(np.mean(np.array([x.to_numpy() for x in Importances["ExIFFI"]["importances"]]),axis=0)).sort_values(by=[0],ascending=False)
    importances_e_diffi_plus=pd.DataFrame(np.mean(np.array([x.to_numpy() for x in Importances["ExIFFI_plus"]["importances"]]),axis=0)).sort_values(by=[0],ascending=False)
    #importances_ediffi_depth=pd.DataFrame(np.mean(np.array([x.to_numpy() for x in Importances["ExDIFFI"]["importances"]]),axis=0)).sort_values(by=[0],ascending=False)
    importances_diffi=pd.DataFrame(np.mean(np.array([x.to_numpy() for x in Importances["DIFFI"]["importances"]]),axis=0)).sort_values(by=[0],ascending=False)
    with open('c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\results\\compare_features\\results\\Importances_davide\\importances_'+name,'wb') as f:
        pickle.dump((importances_e_diffi,importances_e_diffi_plus,importances_diffi), f)
        
def create_robustness_dict(precision,Importances,name):
    Robustness = {"cosine":{"ExIFFI":[],"ExDIFFI":[],"DIFFI":[]},
                  "kendall":{"ExIFFI":[],"ExDIFFI":[],"DIFFI":[]}
                  }
    
    a=np.arange(len(Importances["ExIFFI"]["robustness"][0]))
    b=len(a)-1-a
    min = 1 - cosine_ordered_similarity(a,b)
    for i in range(precision):
        for j in range(i+1,precision):
            result_ediffi = (1 - cosine_ordered_similarity(Importances["ExIFFI"]["robustness"][i], Importances["ExIFFI"]["robustness"][j])-min)/(1-min)
            result_ediffi_depth  = (1 - cosine_ordered_similarity(Importances["ExDIFFI"]["robustness"][i], Importances["ExDIFFI"]["robustness"][j])-min)/(1-min)
            result_diffi  = (1 - cosine_ordered_similarity(Importances["DIFFI"]["robustness"][i], Importances["DIFFI"]["robustness"][j])-min)/(1-min)
            Robustness["cosine"]["ExIFFI"].append(result_ediffi)
            Robustness["cosine"]["ExDIFFI"].append(result_ediffi_depth)
            Robustness["cosine"]["DIFFI"].append(result_diffi)
            
            result_ediffi,p_value = kendalltau(Importances["ExIFFI"]["robustness"][i], Importances["ExIFFI"]["robustness"][j])
            result_ediffi_depth,p_value = kendalltau(Importances["ExDIFFI"]["robustness"][i], Importances["ExDIFFI"]["robustness"][j])
            result_diffi,p_value = kendalltau(Importances["DIFFI"]["robustness"][i], Importances["DIFFI"]["robustness"][j])
            Robustness["kendall"]["ExIFFI"].append(result_ediffi)
            Robustness["kendall"]["ExDIFFI"].append(result_ediffi_depth)
            Robustness["kendall"]["DIFFI"].append(result_diffi)     
    
    path = '../results/compare_features/results/Robustness_dict/'+name+".pkl"
    with open(path, 'wb') as f:
        pickle.dump(Robustness, f)
    return Robustness 
     
def create_comparison_dict(precision,Importances,name):
    results = {"ExIFFI_ExDIFFI":{"cosine":[],"kendall":[]},
           "ExIFFI_DIFFI":{"cosine":[],"kendall":[]},
           "DIFFI_ExDIFFI":{"cosine":[],"kendall":[]}}

    for i in range(precision):
        for j in range(precision):
            cosine_ediffi_ediffi_depth = 1 - cosine_ordered_similarity(Importances["ExIFFI"]["robustness"][i], Importances["ExDIFFI"]["robustness"][j])
            kendall_ediffi_ediffi_depth = kendalltau(Importances["ExIFFI"]["robustness"][i], Importances["ExDIFFI"]["robustness"][j])
            results["ExIFFI_ExDIFFI"]["cosine"].append(cosine_ediffi_ediffi_depth)
            results["ExIFFI_ExDIFFI"]["kendall"].append(kendall_ediffi_ediffi_depth)
            
            cosine_ediffi_diffi = 1 - cosine_ordered_similarity(Importances["ExIFFI"]["robustness"][i], Importances["DIFFI"]["robustness"][j])
            kendall_ediffi_diffi = kendalltau(Importances["ExIFFI"]["robustness"][i], Importances["DIFFI"]["robustness"][j])
            results["ExIFFI_DIFFI"]["cosine"].append(cosine_ediffi_diffi)       
            results["ExIFFI_DIFFI"]["kendall"].append(kendall_ediffi_diffi)
            
            cosine_ediffi_depth_diffi = 1 - cosine_ordered_similarity(Importances["ExDIFFI"]["robustness"][i], Importances["DIFFI"]["robustness"][j])
            kendall_ediffi_depth_diffi = kendalltau(Importances["ExDIFFI"]["robustness"][i], Importances["DIFFI"]["robustness"][j])
            results["DIFFI_ExDIFFI"]["cosine"].append(cosine_ediffi_depth_diffi)       
            results["DIFFI_ExDIFFI"]["kendall"].append(kendall_ediffi_depth_diffi)  
            
    path = '../results/compare_features/results/Comparison_dict/'+name+".pkl"
    with open(path, 'wb') as f:
        pickle.dump(results, f)   
             
    return results

def load_variables(pwd,name):
    path = '../results/compare_features/results/Importances_dict/'+name+".pkl"
    with open(path, 'rb') as f:
        Importances = pickle.load(f)
    path = '../results/compare_features/results/Robustness_dict/'+name+".pkl"
    with open(path, 'rb') as f:
        Robustness = pickle.load(f)
    path = '../results/compare_features/results/Comparison_dict/'+name+".pkl"
    with open(path, 'rb') as f:
        Comparison = pickle.load(f)    
    return Importances,Robustness,Comparison


    
        