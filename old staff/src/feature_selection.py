import pickle 

from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier

from models.Extended_DIFFI_original import *
from models.Extended_IF import * 
#from models.forests import *

#from utils.utils import *
from utils import *

#functions used in the feature selection tasks
            
def compute_feature_selection(importances, dim, X, X_train, y, y_train, precision, n_trees, name, feature_ranking_algorithm):
    """
    Compute and save in pkl files the information needed to obtain the Feature Selection Plot. For each different set of input features compute 
    the Average Precision of the EIF_plus model on multiple executions. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------

    importances: dict
            Dictionary containing the Importance Scores sorted in decreasing order, obtained from the make_importances_file function
    
    dim: int
            Number of features in the dataset

    X: pd.DataFrame
            Input dataset
    
    X_train: pd.DataFrame
            Training Set for the Isolation-based models.
    
    y: np.array
            Input labels

    y_train: np.array 
            Training Set labels

    precision: int
            Number of executions used to compute the Average Precision values

    n_trees: int
            Number of trees to use in the definition of the EIF_plus model
    
    name: string
            Dataset's name

    feature_ranking_algorithm: string
            Name of the Feature Ranking Algorithm used to perform Feature Selection

    Returns
    ----------
    A dictionary containing the Average Precision scores for each different set of input features is saved in a pkl file. 
    """

    # Casual Feature Selection -> remove a random feature at every step 
    if feature_ranking_algorithm=="casual":
        precisionsEIF=[]
        l_o=list(range(dim))
        for i in tqdm(range(0,dim)):
            
            precisionEIF=[]
            for s in range(0,precision): #precision is equal to 10 in compare_feature_selection.py so here it is used 
                #just to do 10 instances of each model 
                EIF=None
                while not EIF:
                    l=np.random.choice(l_o,dim-i,replace=False)
                    X_red_train,y_red_train = X_train[:,l],y_train
                    X_red,y_red = X[:,l],y
                    X_red_train,y_red_train=drop_duplicates(X_red_train,y_red_train)
                    X_red,y_red = drop_duplicates(X_red,y_red)
                    EIF=ExtendedIF(n_trees=n_trees,plus=1)
                    EIF.fit(X_red_train)
                    score = EIF.predict(X_red)
                    avg_prec = sklearn.metrics.average_precision_score(y_red,score)
                precisionEIF.append(avg_prec)
            precisionsEIF.append(precisionEIF)

        #At the end save the result in a pkl file 
        #path = '../results/feature_selection/results/AvgPrecisions/'+ feature_ranking_algorithm + '_' + name
        #New put for Davide exeuctions
        path = 'c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\results\\feature_selection\\results\\Precisions_davide\\'+ feature_ranking_algorithm + '_' + name
        with open(path, 'wb') as f:
            pickle.dump([precisionsEIF], f)  

    #ExIFFI Feature Selection -> Remove every time the least important feature according to the ExIFFI feature importance score
    else:
        precisionsEIF=[]
        #Importances = importances[feature_ranking_algorithm].sort_values(by=[0],ascending=False)
        Importances = importances[feature_ranking_algorithm]
        for i in tqdm(range(0,dim)):
            precisionEIF=[]
            X_red_train,y_red_train=X_train[:,list(Importances.index)[:dim-i]],y_train
            X_red,y_red=X[:,list(Importances.index)[:dim-i]],y
            X_red_train,y_red_train=drop_duplicates(X_red_train,y_red_train)
            X_red,y_red = drop_duplicates(X_red,y_red)
            for s in range(precision):
            
                try:
                    EIF=ExtendedIF(n_trees=n_trees,plus=1)
                    EIF.fit(X_red_train)
                    score = EIF.predict(X_red)
                    avg_prec = sklearn.metrics.average_precision_score(y_red,score)
                except: 
                    avg_prec = np.nan
                precisionEIF.append(avg_prec)
            precisionsEIF.append(precisionEIF)

        #At the end save the result in a pkl file 
        #path = '../results/feature_selection/results/AvgPrecisions/'+ feature_ranking_algorithm + '_' + name
        path = 'c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\results\\feature_selection\\results\\Precisions_davide\\'+ feature_ranking_algorithm + '_' + name
        with open(path, 'wb') as f:
            pickle.dump([precisionsEIF], f)
            
            
            
#Load the results from the pkl files            

def open_precisions(name,feature_ranking_algorithm):
    """
    Load the pkl file obtained from the compute_feature_selection function
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    name: string
            Dataset's name

    feature_ranking_algorithm: string
            Name of the Feature Ranking Algorithm used to perform Feature Selection

    Returns
    ----------
    precisionsEIF: Dictionary with Average Precision values for different input sets of features. 
    """
    #path = '../results/feature_selection/results/AvgPrecisions/'+ feature_ranking_algorithm + '_' + name
    path = 'c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\results\\feature_selection\\results\\Precisions_davide\\'+ feature_ranking_algorithm + '_' + name
    with open(path, "rb") as f:
        [precisionsEIF] = pickle.load(f)  
    return precisionsEIF



def plot_featsel(precisions_dict,name,pwd):
    """
    Obtain the Feature Selection plot exploiting the information collected with the compute_feature_selection function 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    precisions_dict: dict
            Average Precision dicitonary loaded with the open_precisions function 
    
    name: string
            Dataset's name

    pwd: string
            Current working directory

    Returns
    ----------
    Feature Selection Plot. The plot is also saved locally as a PDF on the machine executing the code.  
    """
    colors = ["tab:red","tab:gray","tab:orange","tab:green","tab:blue","tab:olive",'tab:brown']
    #markers = ["P","s","o","v","*","+"]
    c=0
    for i in list(precisions_dict.keys())[-1::-1]:
        median     = [np.percentile(x, 50) for x in precisions_dict[i]]
        five       = [np.percentile(x, 95) for x in precisions_dict[i]]
        ninetyfive = [np.percentile(x, 5) for x in precisions_dict[i]]
        dim = len(median)
        
        #i.replace("E-DIFFI","ExIFFI")
        #i.replace('E-IFFI-plus','ExIFFI-plus')
        #i.replace("random_forest","Random Forest")
        plt.style.use('default')
        plt.rcParams['axes.facecolor'] = '#F2F2F2'
        plt.grid(alpha = 0.7)
        plt.plot(median,label=i,c=colors[c],alpha=0.5,marker="o")#markers[c])
        
        plt.xlabel("Number of Features",fontsize = 20)
        plt.ylabel("Average Precision",fontsize = 20)
        #plt.title("Feature selection "+name, fontsize = 18)
        plt.xticks(range(dim),range(dim,0,-1))
        
        
        
        plt.fill_between(np.arange(dim),five, ninetyfive,alpha=0.05, color=colors[c])
        plt.legend(bbox_to_anchor = (1.05,0.95),loc="upper left")
        plt.grid(visible=True, alpha=0.5, which='major', color='gray', linestyle='-')
        #plt.savefig(pwd+'/results_feature_selection/avgimages/'+name+'_2.pdf',bbox_inches = "tight")
        plt.savefig(pwd+'\\results\\davide\\Featsel Plots\\feature_selection_plots_final\\featsel_' + name + '_final.pdf',bbox_inches = "tight")
        c+=1
    plt.show()
    
    
def Random_Forest_Feature_importance(name):
    """
    Obtain the dataset's input features sorted in decreasing order of importance according to the Random Forest model.
    The hardcoded Series composing the feature_dictionary dictionary were obtained using the compute_rf_feat_imp function.  
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    name: string
            Dataset's name
    Returns
    ----------
    feature_dictionary[name]: pd.Series
            Dataset's input features in decreasing order of importance according to the Random Forest model   
    """
    feature_dictionary={"annthyroid"    :    pd.Series([1,5,3,2,4,0]),
                        "breastw"       :    pd.Series([1,5,2,6,7,4,0,3,8]),
                        "cardio"        :    pd.Series([17,6,7,16,9,18,8,19,11,10,3,13,12,0,1,2,4,14,20,5,15]),
                        "ionosphere"    :    pd.Series([2,25,1,5,2,4,6,26,22,14,29,12,27,31,10,0,16,20,30,8,32,23,19,21,18,24,9,7,11,15,13,17,28]),
                        "lympho"        :    pd.Series([8,10,6,0,3,12,11,13,17,9,7,4,15,2,14,16,5,1]),
                        "mammography"   :    pd.Series([4,3,5,0,1,2]),
                        "pendigits"     :    pd.Series([8,5,4,13,10,2,7,15,11,6,3,14,12,9,1,0]),
                        "pima"          :    pd.Series([1,5,7,6,2,0,3,4]),
                        "satellite"     :    pd.Series([17,21,5,1,33,29,13,9,25,23,19,18,16,20,35,11,31,15,3,30,22,32,28,24,34,7,10,14,26,4,12,0,8,27,2,6]),
                        "shuttle"       :    pd.Series([0,6,8,7,4,1,2,5,3]),
                        "thyroid"       :    pd.Series([5,1,3,2,4,0]),
                        "vertebral"     :    pd.Series([5,4,1,3,0,2]),
                        "vowels"        :    pd.Series([0,7,6,2,9,3,5,1,8,4,10,11]),
                        "wine"          :    pd.Series([12,0,6,5,3,11,4,8,9,2,10,7,1]),
                        "glass"         :    pd.Series([5,0,1,4,3,6,7,2,8]),
                        "diabetes"      :    pd.Series([2,3,0,1]),
                        "moodify"       :    pd.Series([6,3,2,5,8,1,0,7,9,4,10]),
                        }
    return feature_dictionary[name]

def compute_rf_feat_imp(name):
    ''' 
    Obtain the input features of a specific dataset sorted in decreasing order of importance according to the Random Forest model. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    name: string
            Dataset's name
    Returns
    ----------
    RF.feature_importances_.argsort()[::-1]: np.array
            Input features of the input dataset sorted in decreasing order of Feature Importance
    '''        
    os.chdir('c:\\Users\\lemeda98\\Desktop\\PHD Information Engineering\\ExIFFI\\ExIFFI\\data')
    if name=='diabetes' or name=='moodify':
        X,y=csv_dataset(name,os.getcwd()+'\\')
    else:
        X,y=dataset(name,os.getcwd()+'\\')

    X,y=downsample(X,y)
    X_train,X_test=partition_data(X,y)
    scaler=StandardScaler()
    X=np.r_[X_train,X_test]
    X=scaler.fit_transform(X)
    y_train=np.zeros(X_train.shape[0])
    y_test=np.ones(X_test.shape[0])
    y=np.concatenate([y_train,y_test])

    RF=RandomForestClassifier(n_estimators=200)
    RF.fit(X,y)

    return RF.feature_importances_.argsort()[::-1]



def Random_Forest_Feature_importance_scaled(name):
    """
    Obtain the dataset's input features sorted in decreasing order of importance according to the Random Forest model.
    Differently from the Random_Forest_Feature_importance function in this case the importance scores, used to obtain the 
    feature ranking, are obtained scaling the training set before training.
    The hardcoded Series composing the feature_dictionary dictionary were obtained using the compute_rf_feat_imp function.
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    name: string
            Dataset's name
    Returns
    ----------
    feature_dictionary[name]: pd.Series
            Dataset's input features in decreasing order of importance according to the Random Forest model   
    """

    feature_dictionary={
        'wine': pd.Series([12,0,6,3,5,4,9,11,10,8,2,7,1]),
        'annthyroid': pd.Series([1,5,2,3,4,0]),  
        'breastw': pd.Series([1,2,5,6,7,4,0,3,8]),  
        'shuttle': pd.Series([0,6,8,4,7,1,2,3,5]),
        'pima': pd.Series([1,5,7,6,2,0,4,3]),
        'cardio': pd.Series([17,6,7,16,8,9,18,11,10,3,19,13,0,12,1,2,4,
            14,20,5,15]),
        'glass': pd.Series([5,0,1,4,3,6,2,7,8]),
        'ionosphere': pd.Series([3,1,5,25,4,6,2,29,14,26,12,0,27,31,16,22,8,
            32,18,10,20,24,7,21,13,30,23,19,9,28,11,15,17]),
        'pendigits': pd.Series([ 8,5,13,4,10,2,11,15,7,3,6,14,12,9,1,0]),
        'diabetes': pd.Series([2,3,1,0]),
        'moodify': pd.Series([ 6,3,2,5,8,1,0,9,10,7,4])
        
    }

    return feature_dictionary[name]
