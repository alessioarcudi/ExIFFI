import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm
from matplotlib.pyplot import *

from models.interpretability_module import local_diffi

def plt_importances_bars(importances, name, pwd, dim, f = 6):
    """
    Obtain the Global Importance Bar Plot given the Importance Scores values computed in the compute_imps function. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    importances: np.array
            2-dimensional array containing the Global Feature Importance vector for different executions of the ExIFFI model. 
            Obtained from the compute_imps function. 
    name: string
            Dataset's name
    pwd: string
            Current working directory
    dim: int
            Number of features in the input dataset
    f: int
            Number of vertical bars to include in the Bar Plot. By default f is set to 6. 
    Returns
    ----------
    Obtain the Bar Plot which is then saved locally as a PDF.    
    """
    
    if 'GFI_' not in name:
        name='LFI_'+name
    number_colours = 20
    color = plt.cm.get_cmap('tab20',number_colours).colors
    patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    importances_matrix = np.array([np.array(pd.Series(x).sort_values(ascending = False).index).T for x in importances])
    bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
    bars = pd.DataFrame(bars)
    display(bars)

    tick_names=[]
    for i in range(1,f+1):
        if i==1:
            tick_names.append(r'${}'.format(i) + r'^{st}$')
        elif i==2:
            tick_names.append(r'${}'.format(i) + r'^{nd}$')
        elif i==3:
            tick_names.append(r'${}'.format(i) + r'^{rd}$')
        else:
            tick_names.append(r'${}'.format(i) + r'^{th}$')

    barWidth = 0.85
    r=range(dim)
 
    for i in range(dim):
        plt.bar(r[:f], bars.T.iloc[i,:f].values, bottom=bars.T.iloc[:i,:f].sum().values ,color=color[i%number_colours], edgecolor='white', width=barWidth, label=str(i), hatch=patterns[i//number_colours])

    plt.xlabel("Rank", fontsize = 20)
    
    plt.xticks(range(f),tick_names[:f])
    plt.ylabel("Percentage count", fontsize = 20)
    plt.yticks(range(10,101,10),[str(x)+"%" for x in range(10,101,10)])
    plt.legend(bbox_to_anchor = (1.05,0.95),loc="upper left")
    plt.savefig(pwd+'//results//davide/{}_synt.pdf'
                .format(name),bbox_inches='tight')
    plt.show()

def compute_imps(model,X_train,X_test,n_runs,name,pwd,dim,f=6):
    """
    Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
    functions. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    model: Object
            An instance of the ExIFFI model
    X_train: pd.DataFrame
            Training Set 
    X_test: pd.DataFrame
            Test Set 
    n_runs: int
            Number of executions used to collect the data
    name: string
            Dataset's name
    pwd: string
            Current working directory
    dim: int
            Number of features in the input dataset
    f: int
            Number of vertical bars to include in the Bar Plot. By default f is set to 6. 
    Returns
    ----------
    imps: np.array
            2-dimensional array containing the Global Feature Importance vector for different executions of the ExIFFI model.
            The array is also locally saved in a pkl file for the sake of reproducibility.
    plt_data: dict
            Dictionary containig the average Importance Scores values, the feature order and the standard deviations on the Importance
            Scores.
            The dictionary is also locally saved in a pkl file for the sake of reproducibility.
    """

    name='GFI_'+name
    imps=np.zeros(shape=(n_runs,X_train.shape[1]))
    for i in tqdm(range(n_runs)):
        model.fit(X_train)
        imps[i,:]=model.Global_importance(X_test,calculate=True,sovrascrivi=False,depth_based=False)

    path = pwd + '\\results\\davide\\Importance_Scores\\Imp Score\\imp_score_' + name + '.pkl'
    with open(path, 'wb') as fl:
        pickle.dump(imps,fl)

    #Take the mean feature importance scores over the different runs for the Feature Importance Plot
    #and put it in decreasing order of importance
    mean_imp=np.mean(imps,axis=0)
    std_imp=np.std(imps,axis=0)
    mean_imp_val=np.sort(mean_imp)
    feat_order=mean_imp.argsort()

    plt_data={'Importances': mean_imp_val,
              'feat_order': feat_order,
              'std': std_imp[mean_imp.argsort()]}

    path = pwd + '\\results\\davide\\Importance_Scores\\Plt Data\\plt_data_' + name + '.pkl'
    with open(path, 'wb') as fl:
        pickle.dump(plt_data,fl)

    return imps,plt_data

def plt_feat_bar_plot(global_importances,name,pwd,f=6,save=True):
    """
    Obtain the Global Feature Importance Score Plot exploiting the information obtained from compute_imps function. 
    --------------------------------------------------------------------------------
    
    Parameters
    ----------
    global_importances: np.array
            Average Global Importance values across multiple executions of the ExIFFI model. 
    name: string
            Dataset's name
    pwd: string
            Current working directory
    f: int
            Number of vertical bars to include in the Bar Plot. By default f is set to 6. 
    save: bool 
            Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. 
    Returns
    ----------
    Obtain the Score Plot which is also locally saved as a PDF. 
    """
    
    name_file='Feat_bar_plot_'+name 
    patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

    imp_vals=global_importances['Importances']

    feat_imp=pd.DataFrame({'Global Importance': np.round(imp_vals,3),
                          'Feature': global_importances['feat_order'],
                          'std': global_importances['std']
                          })

    if len(feat_imp)>15:
        feat_imp=feat_imp.iloc[-15:].reset_index(drop=True)
    
    dim=feat_imp.shape[0]

    number_colours = 20

    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = '#F2F2F2'
    plt.rcParams['axes.axisbelow'] = True
    color = plt.cm.get_cmap('tab20',number_colours).colors
    ax1=feat_imp.plot(y='Global Importance',x='Feature',kind="barh",color=color[feat_imp['Feature']%number_colours],xerr='std',
                     capsize=5, alpha=1,legend=False,
                     hatch=[patterns[i//number_colours] for i in feat_imp['Feature']])
    xlim=np.min(imp_vals)-0.2*np.min(imp_vals)

    ax1.grid(alpha=0.7)
    ax2 = ax1.twinx()
    # Add labels on the right side of the bars
    values=[]
    for i, v in enumerate(feat_imp['Global Importance']):
        values.append(str(v) + ' +- ' + str(np.round(feat_imp['std'][i],2)))
    
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(range(dim))
    ax2.set_yticklabels(values)
    ax2.grid(alpha=0)
    plt.axvline(x=0, color=".5")
    ax1.set_xlabel('Importance Score',fontsize=20)
    ax1.set_ylabel('Features',fontsize=20)
    plt.xlim(xlim)
    plt.subplots_adjust(left=0.3)
    if save:
        plt.savefig(pwd+'//results//davide/{}.pdf'.format(name_file),bbox_inches='tight')
        
    plt.show()

def plt_global_importance_bars(name,pwd,dim,f=6):
    """
    Produce the Bar Plot and the Score Plot using the data obtained from the compute_imps function.  
    --------------------------------------------------------------------------------
    
    Parameters
    ---------- 
    name: string
            Dataset's name
    pwd: string
            Current working directory
    dim: int
            Number of features in the input dataset
    f: int
            Number of vertical bars to include in the Bar Plot. By default f is set to 6. 
    Returns
    ----------
    Obtain the Score Plot which is also locally saved as a PDF. 
    """

    path = pwd + '\\results\\davide\\Importance_Scores\\Imp Score\\imp_score_GFI_' + name + '.pkl'
    with open(path, 'rb') as fl:
        imps = pickle.load(fl)

    path = pwd + '\\results\\davide\\Importance_Scores\\Plt Data\\plt_data_GFI_' + name + '.pkl'
    with open(path, 'rb') as fl:
        plt_data = pickle.load(fl)

    plt_importances_bars(imps,name,pwd,dim)
    plt_feat_bar_plot(plt_data,name,pwd)
    


def plot_importance_map(name,model,X_train,y_train,resolution,pwd,save=True,m=None,factor=3,feats_plot=(0,1),plt=plt,isdiffi=False):
    """
    Produce the Local Feature Importance Scoremap.   
    --------------------------------------------------------------------------------
    
    Parameters
    ---------- 
    name: string
            Dataset's name
    model: Object
            Instance of the ExIFFI model. 
    X_train: pd.DataFrame
            Training Set 
    y_train: np.array
            Dataset training labels
    resolution: int
            Scoremap resolution 
    pwd: string
            Current working directory
    save: bool 
            Boolean variable used to decide weather to save the Score Plot locally as a PDF or not.
    m: bool
            Boolean variable regulating the plt.pcolor advanced settings. By defualt the value of m is set to None
    factor: int
            Integer factor used to define the minimum and maximum value of the points used to create the scoremap. 
            By default the value of f is set to 3.
    feats_plot: tuple
            This tuple contains the indexes of the pair features to compare in the Scoremap. By default the value of feats_plot
            is set to (0,1)
    plt: Object
            Plt object used to create the plot. 
    isdiffi:bool
            Boolean variable used to decide wether to create the Scoremap using the ExIFFI Local Importance Score or 
            the Local Importance Scores of the DIFFI model. By default the isdiffi value is set to False. 
    Returns
    ----------
    Obtain the Scoremap which is also locally saved as a PDF. 
    """
    mins = X_train.min(axis=0)[list(feats_plot)]
    maxs = X_train.max(axis=0)[list(feats_plot)]  
    mean = X_train.mean(axis = 0)
    mins = list(mins-(maxs-mins)*factor/10)
    maxs = list(maxs+(maxs-mins)*factor/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
    mean = np.repeat(np.expand_dims(mean,0),len(xx)**2,axis = 0)
    mean[:,feats_plot[0]]=xx.reshape(len(xx)**2)
    mean[:,feats_plot[1]]=yy.reshape(len(yy)**2)

    importance_matrix = np.zeros_like(mean)
    if isdiffi:
        model.max_samples = len(X_train)
        for i in range(importance_matrix.shape[0]):
            importance_matrix[i] = local_diffi(model, mean[i])[0]
    else:
        importance_matrix = model.Local_importances(mean, True, False)
    
    sign = np.sign(importance_matrix[:,feats_plot[0]]-importance_matrix[:,feats_plot[1]])
    Score = sign*((sign>0)*importance_matrix[:,feats_plot[0]]+(sign<0)*importance_matrix[:,feats_plot[1]])
    x = X_train[:,feats_plot[0]].squeeze()
    y = X_train[:,feats_plot[1]].squeeze()
    
    Score = Score.reshape(xx.shape)

    if m is not None:
        cp = plt.pcolor(xx, yy, Score, cmap=cm.RdBu, vmin=-m, vmax=m, shading='nearest')
    else:
        cp = plt.pcolor(xx, yy, Score , cmap=cm.RdBu, shading='nearest',norm=colors.CenteredNorm())
    plt.contour(xx, yy, (importance_matrix[:,feats_plot[0]]+importance_matrix[:,feats_plot[1]]).reshape(xx.shape),levels = 7, cmap=cm.Greys,alpha=0.7)
    plt.scatter(x[y_train==0],y[y_train==0],s=40,c="tab:blue",marker="o",edgecolors="k",label="inliers")
    plt.scatter(x[y_train==1],y[y_train==1],s=60,c="tab:orange",marker="*",edgecolors="k",label="outliers")
    plt.legend()
    if save:
        plt.savefig(pwd+'\\results\\davide\\Local Scoremaps Plot\\Local_Importance_Scoremap_{}.pdf'
                .format(name),bbox_inches='tight')
    plt.show()

def print_score_map(model,X,resolution):
    """
    Produce the Anomaly Score Scoremap.   
    --------------------------------------------------------------------------------
    
    Parameters
    ---------- 
    model: Object
            Instance of the ExIFFI model. 
    X: pd.DataFrame
            Input dataset
    resolution: int
            Scoremap resolution 
    Returns
    ----------
    Returns the Anomaly Score Scoremap  
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    mins = list(mins-(maxs-mins)*3/10)
    maxs = list(maxs+(maxs-mins)*3/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))

    S1 = model.Anomaly_Score(X_in=np.c_[xx.ravel(), yy.ravel()])
    S1 = S1.reshape(xx.shape)
    x= X.T[0]
    y= X.T[1]        

    plt.figure(figsize=(12,12)) 
    levels = np.linspace(np.min(S1),np.max(S1),10)
    CS = plt.contourf(xx, yy, S1, levels, cmap=plt.cm.YlOrRd)
    plt.scatter(x,y,s=15,c='None',edgecolor='k')
    plt.axis("equal")
    plt.show()
    return


''' 
def plt_bars(Importances, imp_alg, name, dim, f = 6):
    number_colours = 20
    color = plt.cm.get_cmap('tab20',number_colours).colors
    patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    by=0
    importances_matrix = np.c_[[np.array(x.sort_values(by = [by],ascending = False).index).T for x in Importances[imp_alg]["importances"]]]
    bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
    
    bars = pd.DataFrame(bars)
                    
    barWidth = 0.85
    r=range(dim)

    for i in range(dim):
        plt.bar(r[:6], bars.T.iloc[i,:f].values, bottom=bars.T.iloc[:i,:f].sum().values ,color=color[i%number_colours], edgecolor='white', width=barWidth, label=str(i), hatch=patterns[i//number_colours])

    if imp_alg=="E-diffi" or imp_alg=="E-DIFFI":
        imp_alg="ExIFFI"
    elif imp_alg=="diffi":
        imp_alg="DIFFI"
    
    name = name.capitalize()
    plt.xlabel("rank", fontsize = 14)
    plt.xticks(range(6),[r'$1^{st}$', r'$2^{nd}$', r'$3^{rd}$', r'$4^{th}$', r'$5^{th}$', r'$6^{th}$', r'$7^{th}$', r'$8^{th}$', r'$9^{th}$'][:f])
    plt.ylabel("Percentage count", fontsize = 14)
    plt.yticks(range(10,101,10),[str(x)+"%" for x in range(10,101,10)])
    if dim >6:
        plt.title("{} {} \n percentage count of feature ranking \n in the first {} position".format(name,imp_alg,f), fontsize = 18)
    else:
        plt.title("{} {} \n percentage count of feature ranking".format(name,imp_alg,f), fontsize = 18)        
    plt.legend(bbox_to_anchor = (1.05,0.95),loc="upper left",ncol=2)
    plt.savefig('./results/compare_features/images/'+imp_alg+'_'+name+'.pdf',bbox_inches = "tight")
    plt.show()
'''

''' 
def plot_importance_map_2(EDIFFI, X_train, y_train, risoluzione, m=None,feats_plot = (0,1), plt=plt):
    mins = X_train.min(axis=0)[list(feats_plot)]
    maxs = X_train.max(axis=0)[list(feats_plot)]  
    mean = X_train.mean(axis = 0)
    mins = list(mins-(maxs-mins)*3/10)
    maxs = list(maxs+(maxs-mins)*3/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], risoluzione), np.linspace(mins[1], maxs[1], risoluzione))
    mean = np.repeat(np.expand_dims(mean,0),len(xx)**2,axis = 0)
    mean[:,feats_plot[0]]=xx.reshape(len(xx)**2)
    mean[:,feats_plot[1]]=yy.reshape(len(yy)**2)
    
    importance_matrix = EDIFFI.Local_importances(mean, True, False)

    x = X_train[:,feats_plot[0]].squeeze()
    y = X_train[:,feats_plot[1]].squeeze()
    
    R=importance_matrix[:,feats_plot[0]].reshape(xx.shape)
    B=importance_matrix[:,feats_plot[1]].reshape(xx.shape)
    B=(B-np.min(B))/(m-np.min(B))
    B[B>1]=1
    R=(R-np.min(R))/(m-np.min(R))
    R[R>1]=1

    def screen(B,x,y,z):
        return np.array([B*x,B*y,B*z])

    R_screen = screen(R,0.88,0.88,0.50)
    B_screen = screen(B,0.50,0.88,0.88)

    image = 1-(R_screen**2+B_screen**2)/1.2

    plt.imshow(image.transpose(1,2,0), extent=[mins[0],maxs[0],mins[1],maxs[1]], origin='lower')
    plt.scatter(x,y,s=15,alpha=(y_train+1)/2,c=y_train,cmap=plt.cm.PuOr)
    plt.show()
'''