from __future__ import annotations

import os
import time
from typing import Type, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, ScalarFormatter
import seaborn as sns 
sns.set()
import pandas as pd
import pickle
from collections import namedtuple

from utils_reboot.datasets import Dataset
from utils_reboot.utils import open_element,get_most_recent_file
from utils_reboot.experiments import compute_plt_data
from model_reboot.EIF_reboot import ExtendedIsolationForest
from matplotlib import colors, cm
from sklearn.ensemble import IsolationForest
from model_reboot.interpretability_module import local_diffi


def bar_plot(dataset: Type[Dataset], 
            global_importances_file: str,
            filetype: str = "npz", 
            plot_path: str = os.getcwd(), 
            f: int = 6, 
            save_image = True, 
            show_plot = True,
            model:str='EIF+',
            interpretation:str="EXIFFI+",
            scenario:int=1) -> tuple[plt.figure, plt.axes, pd.DataFrame]:
    """
    Compute the Global Importance Bar Plot starting from the Global Feature Importance vector.  
    
    Args:
        dataset: Input dataset
        global_importances_file: The path to the file containing the global importances.
        filetype: The file type of the global importances file. Defaults to "npz".
        plot_path: The path where the plot will be saved. Defaults to os.getcwd().
        f: The number of ranks to be displayed in the plot. Defaults to 6. 
        save_image: A boolean indicating whether the plot should be saved. Defaults to True.
        show_plot: A boolean indicating whether the plot should be displayed. Defaults to True.
        model: The AD model on which the importances should be computed. Defaults to 'EIF+'.
        interpretation: The interpretation model used. Defaults to 'EXIFFI+'.
        scenario: The scenario number. Defaults to 1.

    Returns:
       The figure, the axes and the bars dataframe.   
    """
    
    if  isinstance(dataset.feature_names, np.ndarray):
        col_names = dataset.feature_names.astype(str)
    elif isinstance(dataset.feature_names, list):
        col_names = dataset.feature_names
    
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    
    if (model=='EIF+' and interpretation=='EXIFFI+') or (model=='EIF' and interpretation=='EXIFFI'):
        name_file = f"{current_time}_GFI_Bar_plot_{dataset.name}_{interpretation}_{scenario}"
    else:
        name_file = f"{current_time}_GFI_Bar_plot_{dataset.name}_{model}_{interpretation}_{scenario}"
    
    #Load the imps array from the pkl file contained in imps_path -> the imps_path is returned from the 
    #compute_local_importances or compute_global_importances functions so we have it for free 
    try:
        importances = open_element(global_importances_file, filetype=filetype)
    except:
        raise Exception("The file path is not valid")

    number_colours = 20
    color = plt.cm.get_cmap('tab20',number_colours).colors
    patterns=[None,'!','@','#','$','^','&','*','°','(',')','-','_','+','=','[',']','{','}',
    '|',';',':',',','.','<','>','/','?','`','~','\\','!!','@@','##','$$','^^','&&','**','°°','((']
    importances_matrix = np.array([np.array(pd.Series(x).sort_values(ascending = False).index).T for x in importances])
    dim=int(importances.shape[1])

    bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
    bars = pd.DataFrame(bars)

    tick_names=[]
    for i in range(1,f+1):
        if int(str(i)[-1])==1 and (len(str(i))==1 or int(str(i)[-2])!=1):
            tick_names.append(r'${}'.format(i) + r'^{st}$')
        elif int(str(i)[-1])==2 and (len(str(i))==1 or int(str(i)[-2])!=1):
            tick_names.append(r'${}'.format(i) + r'^{nd}$')
        elif int(str(i)[-1])==3 and (len(str(i))==1 or int(str(i)[-2])!=1):
            tick_names.append(r'${}'.format(i) + r'^{rd}$')
        else:
            tick_names.append(r'${}'.format(i) + r'^{th}$')

    barWidth = 0.85
    r = range(dim)
    ncols=1
    if importances.shape[1]>15:
        ncols=2
    elif importances.shape[1]>30:
        ncols=3
    elif importances.shape[1]>45:
        ncols=4
    elif importances.shape[1]>60:
        ncols=5
    elif importances.shape[1]>75:
        ncols=6

    fig, ax = plt.subplots()

    for i in range(dim):
        if col_names is not None: 
            ax.bar(r[:f], bars.T.iloc[i, :f].values, bottom=bars.T.iloc[:i, :f].sum().values, color=color[i % number_colours], edgecolor='white', width=barWidth, label=col_names[i], hatch=patterns[i // number_colours])
        else:
            ax.bar(r[:f], bars.T.iloc[i, :f].values, bottom=bars.T.iloc[:i, :f].sum().values, color=color[i % number_colours], edgecolor='white', width=barWidth, label=str(i), hatch=patterns[i // number_colours])

    ax.set_xlabel("Rank", fontsize=20)
    ax.set_xticks(range(f), tick_names[:f])
    ax.set_ylabel("Percentage count", fontsize=20)
    ax.set_yticks(range(10, 101, 10), [str(x) + "%" for x in range(10, 101, 10)])
    ax.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left",ncol=ncols)
    
    if save_image:
        plt.savefig(plot_path + f'/{name_file}.pdf', bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig, ax, bars
    
    

def score_plot(dataset: Type[Dataset], 
            global_importances_file: str,
            plot_path: str = os.getcwd(), 
            save_image = True, 
            show_plot = True,
            model:str='EIF+',
            interpretation:str="EXIFFI",
            scenario=1) -> tuple[plt.axes, plt.axes]:
    """
    Obtain the Global Feature Importance Score Plot starting from the Global Feature Importance vector.

    Args:
        dataset: Input dataset
        global_importances_file: The path to the file containing the global importances.
        plot_path: The path where the plot will be saved. Defaults to os.getcwd().
        save_image: A boolean indicating whether the plot should be saved. Defaults to True.
        show_plot: A boolean indicating whether the plot should be displayed. Defaults to True.
        model: The AD model on which the importances should be computed. Defaults to 'EIF+'.
        interpretation: The interpretation model used. Defaults to 'EXIFFI'.
        scenario: The scenario number. Defaults to 1.
    
    Returns:
        The two axes objects used to create the plot.

    """
   # Compute the plt_data with the compute_plt_data function
    col_names = dataset.feature_names
    plt_data = compute_plt_data(global_importances_file)

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    if (model=='EIF+' and interpretation=='EXIFFI+') or (model=='EIF' and interpretation=='EXIFFI'):
        name_file = f"{current_time}_GFI_Score_plot_{dataset.name}_{interpretation}_{scenario}"
    else:
        name_file = f"{current_time}_GFI_Score_plot_{dataset.name}_{model}_{interpretation}_{scenario}"

    patterns=[None,'!','@','#','$','^','&','*','°','(',')','-','_','+','=','[',']','{','}',
    '|',';',':',',','.','<','>','/','?','`','~','\\','!!','@@','##','$$','^^','&&','**','°°','((']
    imp_vals=plt_data['Importances']
    feat_imp=pd.DataFrame({'Global Importance': np.round(imp_vals,3),
                        'Feature': plt_data['feat_order'],
                        'std': plt_data['std']
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
    xlim=np.min(imp_vals)-0.05*np.min(imp_vals)

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

    if col_names is not None:
        ax1.set_yticks(range(dim))
        idx=list(feat_imp['Feature'])
        yticks=[col_names[i] for i in idx]
        ax1.set_yticklabels(yticks)

    if save_image:
        plt.savefig(plot_path + f'/{name_file}.pdf', bbox_inches='tight')

    if show_plot:
        plt.show()
        
    return ax1,ax2

def plot_feature_selection(
        precision_file: str,
        plot_path:str,
        precision_file_random:Optional[str]=None,
        color:int=0,
        model:Optional[str]=None,
        eval_model:Optional[str]='EIF+',
        interpretation:Optional[str]=None,
        scenario:Optional[int]=2,
        save_image:bool=True,
        plot_image:bool=False,
        box_loc:tuple=None,
        change_box_loc:float=0.9,
        rotation:bool=False,
        change_ylim:bool=False)-> None:
    
    """
    Obtain the feature selection plot.

    Args:
        precision_file: The path to the file containing the precision values.
        plot_path: The path where the plot will be saved.
        precision_file_random: The path to the file containing precision values computed with the random Feature Selection approach. Defaults to None.
        color: The color of the plot. Defaults to 0.
        model: Name of the AD model. Defaults to None.
        eval_model: Name of the evaluation model. Defaults to 'EIF+'.
        interpretation: Name of the interpretation model used. Defaults to None.
        scenario: The scenario number. Defaults to 2.
        save_image: A boolean indicating whether the plot should be saved. Defaults to True.
        plot_image: A boolean indicating whether the plot should be displayed. Defaults to False.
        box_loc: The location of the text box containing the Area under the curve of Feature Selection value. Defaults to None.
        change_box_loc: Change the y axis value of the text box location containing the Area under the curve of Feature Selection value. Defaults to 0.9.
        rotation: A boolean indicating whether the x ticks should be rotated by 45 degrees. Defaults to False.
        change_ylim: A boolean indicating whether the y axis limits should be changed (from 1 to 1.1). Defaults to False.

    Returns:
        The function saves the plot in the specified path and displays it if the plot_image parameter is set to True.

    """
    
    colors = ["tab:red","tab:gray","tab:orange","tab:green","tab:blue","tab:olive",'tab:brown']
    if model is None:
        model = ""
    #Precisions = namedtuple("Precisions",["direct","inverse","dataset","model","value"])

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    precision = open_element(precision_file)

    #model = precision.model
    aucfs = precision.aucfs

    median_direct     = [np.percentile(x, 50) for x in precision.direct]
    five_direct       = [np.percentile(x, 95) for x in precision.direct]
    ninetyfive_direct = [np.percentile(x, 5) for x in precision.direct]
    median_inverse    = [np.percentile(x, 50) for x in precision.inverse]
    five_inverse       = [np.percentile(x, 95) for x in precision.inverse]
    ninetyfive_inverse = [np.percentile(x, 5) for x in precision.inverse]
    dim = len(median_direct)
    
    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = '#F2F2F2'
    plt.grid(alpha = 0.7)

    #import ipdb; ipdb.set_trace()

    if precision_file_random is not None:
        precision_random=open_element(precision_file_random)
        median_random = [np.percentile(x, 50) for x in precision_random.random]
        plt.plot(median_random,label="random",c=colors[3],alpha=0.5,marker="o")

    plt.plot(median_direct,label="direct",c=colors[4],alpha=0.5,marker="o")#markers[c])
    plt.plot(median_inverse,label="inverse",c=colors[color],alpha=0.5,marker="o")
    
    plt.xlabel("Number of Features",fontsize = 20)
    plt.ylabel("Average Precision",fontsize = 20)
    #plt.title("Feature selection "+model, fontsize = 18)

    if rotation:
        plt.xticks(range(dim),range(dim,0,-1),rotation=45)
    else:
        plt.xticks(range(dim),range(dim,0,-1))    
    
    if box_loc is None:
       box_loc = (len(precision.direct)/2,change_box_loc)

    text_box_content = r'${}'.format("AUC") + r'_{FS}$' + " = " + str(np.round(aucfs,3))
    plt.text(box_loc[0],box_loc[1], text_box_content, bbox=dict(facecolor='white', alpha=0.5, boxstyle="round", pad=0.5), 
         verticalalignment='top', horizontalalignment='right')
    
    if change_ylim:
        plt.ylim(0,1.1)
    else:
        plt.ylim(0,1)

    plt.fill_between(np.arange(dim),five_direct, ninetyfive_direct,alpha=0.1, color="k")
    plt.fill_between(np.arange(dim),five_inverse, ninetyfive_inverse,alpha=0.1, color="k")
    plt.fill_between(np.arange(dim),median_direct, median_inverse,alpha=0.7, color=colors[color])
    plt.legend(bbox_to_anchor = (1.05,0.95),loc="upper left")
    plt.grid(visible=True, alpha=0.5, which='major', color='gray', linestyle='-')

    if model=='EIF+' and interpretation=='EXIFFI+':
        namefile = "/" + current_time + "_" + precision.dataset + "_" + eval_model + '_' + "EXIFFI+" + "_feature_selection_" + str(scenario) + ".pdf"
    elif model=='EIF' and interpretation=='EXIFFI':
            namefile = "/" + current_time + "_" + precision.dataset + "_" + eval_model + '_' + 'EXIFFI' + "_feature_selection_" + str(scenario) + ".pdf"
    else:
        namefile = "/" + current_time + "_" + precision.dataset + "_" + eval_model + '_' + model + "_" + interpretation + "_feature_selection_" + str(scenario) + ".pdf"
    
    if save_image:
        plt.savefig(plot_path+namefile,bbox_inches = "tight")
    if plot_image:
        plt.show()
        

def plot_precision_over_contamination(precisions:np.ndarray,
                                      dataset_name:str,
                                      model_name:str,
                                      plot_path:str,
                                      contamination:np.ndarray=np.linspace(0.0,0.1,10),
                                      save_image:bool=True,
                                      plot_image:bool=False,
                                      ylim:tuple=(0,1)) -> None:
    
    """
    Obtain the precision over contamination plot.

    Args:
        precisions: The precision values for different contamination values, obtained from the contamination_in_training_precision_evaluation method.
        dataset_name: The dataset name.
        model_name: The model name.
        plot_path: The path where the plot will be saved.
        contamination: The contamination values. Defaults to np.linspace(0.0,0.1,10).
        save_image: A boolean indicating whether the plot should be saved. Defaults to True.
        plot_image: A boolean indicating whether the plot should be displayed. Defaults to False.
        ylim: The y axis limits. Defaults to (0,1).

    Returns:
        The function saves the plot in the specified path and displays it if the plot_image parameter is set to True.

    """

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    
    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = '#F2F2F2'
    plt.grid(alpha = 0.7)
    plt.plot(contamination,precisions.mean(axis=1),marker="o",c="tab:blue",alpha=0.5,label=model_name)
    plt.fill_between(contamination, [np.percentile(x, 10) for x in precisions], [np.percentile(x, 90) for x in precisions],alpha=0.1, color="tab:blue")
    
    plt.ylim(ylim)
        
    plt.xlabel("Contamination",fontsize = 20)
    plt.ylabel("Average Precision",fontsize = 20)

    namefile = current_time + "_" + dataset_name + '_' + model_name + "_precision_over_contamination.pdf"
    
    if save_image:
        plt.savefig(plot_path + "/" + namefile, bbox_inches = "tight")
    
    if plot_image:
        plt.show()

def get_contamination_comparison(model1:str,
                            model2:str,
                            dataset_name:str,
                            path:str=os.getcwd()):
    
    """
    Obtain the difference in precision between two models for different contamination values.

    Args:
        model1: The first model name.
        model2: The second model name.
        dataset_name: The dataset name.
        path: Starting path to retrieve the path where the precisions of the two models are stored. Defaults to os.getcwd().
    """
    
    path_model1=path+'/results/'+ dataset_name +'/experiments/contamination/'+model1
    path_model2=path+'/results/'+ dataset_name +'/experiments/contamination/'+model2

    precisions_model1=open_element(get_most_recent_file(path_model1),filetype='pickle')[0]
    precisions_model2=open_element(get_most_recent_file(path_model2),filetype='pickle')[0]
    precisions=precisions_model1-precisions_model2

    return precisions

def importance_map(dataset: Type[Dataset],
                   model: Type[ExtendedIsolationForest],
                   resolution: Optional[int] = 30,
                   path_plot: Optional[str] = os.getcwd(),
                   save_plot: Optional[bool] = True,
                   show_plot: Optional[bool] = False,
                   factor: Optional[int] = 3, 
                   feats_plot: Optional[tuple] = (0,1),
                   col_names: List[str] = None,
                   isdiffi: Optional[bool] = False,
                   scenario: Optional[int] = 2,
                   interpretation: Optional[str] = "EXIFFI+"
                   ) -> None:
        """
        Produce the Local Feature Importance Scoremap.   
        
        Args:
            dataset: Input dataset
            model: The AD model.
            resolution: The resolution of the plot. Defaults to 30.
            path_plot: The path where the plot will be saved. Defaults to os.getcwd().
            save_plot: A boolean indicating whether the plot should be saved. Defaults to True.
            show_plot: A boolean indicating whether the plot should be displayed. Defaults to False.
            factor: The factor by which the min and max values of the features are extended. Defaults to 3.
            feats_plot: The features to be plotted. Defaults to (0,1).
            col_names: The names of the features. Defaults to None.
            isdiffi: A boolean indicating whether the local-DIFFI method should be used to compute the importance values. Defaults to False.
            scenario: The scenario number. Defaults to 2.
            interpretation: Name of the interpretation model used. Defaults to "EXIFFI+".
        
        Returns:
            The function saves the plot in the specified path and displays it if the show_plot parameter is set to True.
        """
        
        mins = dataset.X_test.min(axis=0)[list(feats_plot)]
        maxs = dataset.X_test.max(axis=0)[list(feats_plot)]  
        mean = dataset.X_test.mean(axis = 0)
        mins = list(mins-(maxs-mins)*factor/10)
        maxs = list(maxs+(maxs-mins)*factor/10)
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
        mean = np.repeat(np.expand_dims(mean,0),len(xx)**2,axis = 0)
        mean[:,feats_plot[0]]=xx.reshape(len(xx)**2)
        mean[:,feats_plot[1]]=yy.reshape(len(yy)**2)

        importance_matrix = np.zeros_like(mean)
        if isdiffi:
                model.max_samples = len(dataset.X)
                for i in range(importance_matrix.shape[0]):
                        importance_matrix[i] = local_diffi(model, mean[i])[0]
        else:
            importance_matrix = model.local_importances(mean)
        
        sign = np.sign(importance_matrix[:,feats_plot[0]]-importance_matrix[:,feats_plot[1]])
        Score = sign*((sign>0)*importance_matrix[:,feats_plot[0]]+(sign<0)*importance_matrix[:,feats_plot[1]])
        x = dataset.X_test[:,feats_plot[0]].squeeze()
        y = dataset.X_test[:,feats_plot[1]].squeeze()
        
        Score = Score.reshape(xx.shape)

        # Create a new pyplot object if plt is not provided
        fig, ax = plt.subplots()
        
        cp = ax.pcolor(xx, yy, Score, cmap=cm.RdBu, shading='nearest', norm=colors.CenteredNorm())
        
        ax.contour(xx, yy, (importance_matrix[:, feats_plot[0]] + importance_matrix[:, feats_plot[1]]).reshape(xx.shape), levels=7, cmap=cm.Greys, alpha=0.7)

        try:
            ax.scatter(x[dataset.y_test == 0], y[dataset.y_test == 0], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
            ax.scatter(x[dataset.y_test == 1], y[dataset.y_test == 1], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")
        except IndexError:
            print('Handling the IndexError Exception...')
            ax.scatter(x[(dataset.y_test == 0)[:, 0]], y[(dataset.y_test == 0)[:, 0]], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
            ax.scatter(x[(dataset.y_test == 1)[:, 0]], y[(dataset.y_test == 1)[:, 0]], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")
        
        if (isinstance(col_names, np.ndarray)) or (col_names is None):
            ax.set_xlabel(f'Feature {feats_plot[0]}',fontsize=20)
            ax.set_ylabel(f'Feature {feats_plot[1]}',fontsize=20)
        elif col_names is not None:
            ax.set_xlabel(col_names[feats_plot[0]],fontsize=20)
            ax.set_ylabel(col_names[feats_plot[1]],fontsize=20)
        
        ax.legend()

        
        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        if isdiffi:
            filename = current_time+"_importance_map_"+dataset.name+"_"+model.name+"_"+interpretation+f"_{str(scenario)}"+f"_feat_{feats_plot[0]}_{feats_plot[1]}"+".pdf"
        else:
            filename = current_time+"_importance_map_"+dataset.name+"_"+model.name+"_"+interpretation+f"_{str(scenario)}"+f"_feat_{feats_plot[0]}_{feats_plot[1]}"+".pdf"

        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(path_plot + '/{}'.format(filename), bbox_inches='tight')


def gfi_over_contamination(importances, contamination, model_index, plot_path,col_names=None, save_plot=True, show_plot=False):
    importances_mean = importances[model_index].mean(axis=0)
    importances_95_upper = np.percentile(importances[model_index], 95, axis=0)
    importances_95_lower = np.percentile(importances[model_index], 5, axis=0)
    if col_names is None:
        col_names = [f'Feature {i}' for i in range(importances_mean.shape[1])]
    
    number_colours = 20
    patterns=[None,'!','@','#','$','^','&','*','°','(',')','-','_','+','=','[',']','{','}',
    '|',';',':',',','.','<','>','/','?','`','~','\\','!!','@@','##','$$','^^','&&','**','°°','((']
    color = plt.cm.get_cmap('tab20',number_colours).colors
    for i in range(importances_mean.shape[1]):
        plt.plot(contamination, importances_mean[:,i], color=color[i % number_colours],  label=col_names[i],marker="o")
        #plt.fill_between(contamination, importances_95_lower[:,i], importances_95_upper[:,i], alpha=0.1, color=color[i % number_colours])

    plt.grid(alpha=0)
    plt.xlabel("Contamination", fontsize=20)
    plt.ylabel("Feature importance score", fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left")

        
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    if  save_plot:
        plt.savefig(plot_path + '/'+ current_time + 'gfi_over_contamination_model_contamination=' +str(contamination[model_index]) + '.pdf', bbox_inches='tight')
    if show_plot:
        plt.show()


def get_time_scaling_files(dataset: Type[Dataset],
                           model: Type[ExtendedIsolationForest],
                           experiment_path: str=os.getcwd(),
                           interpretation:str='NA'):    

    path_fit_predict=os.path.join(experiment_path,dataset.name,'experiments','time_scaling',model,'fit_predict')
    fit_pred_times=open_element(get_most_recent_file(path_fit_predict),filetype='pickle')
    if interpretation == "NA":
        return fit_pred_times
    else: 
        path_imp=os.path.join(experiment_path,dataset.name,'experiments','time_scaling',model,interpretation)
        imp_time=open_element(get_most_recent_file(path_imp),filetype='pickle')
        return fit_pred_times,imp_time
    

def get_vals(model: str, 
            dataset_names: List[str],
            type:str='predict') -> tuple[List,List,List]:
    
    """
    Obtain statistics on the execution time of a model for different datasets. These values will be used in the plot_time_scaling method.

    Args:
        model: The model name.
        dataset_names: The list of dataset names.
        type: The type of execution time. Defaults to 'predict'.

    Returns:
       The median, 5th percentile and 95th percentile values of the execution time.
    """

    assert type in ['predict','fit','importances'], "Type not valid"
    
    os.chdir('../utils_reboot')
    with open(os.getcwd() + "/time_scaling_test_dei_new.pickle", "rb") as file:
        dict_time = pickle.load(file)

    val_times=[]
    for d_name in dataset_names:
        time=np.array(dict_time[type][model][d_name])
        val_times.append(time)

    median_val_times=[np.percentile(x,50) for x in val_times]
    five_val_times=[np.percentile(x,5) for x in val_times]
    ninefive_val_times=[np.percentile(x,95) for x in val_times]

    return median_val_times,five_val_times,ninefive_val_times

def plot_time_scaling(model_names:List[str],
                      dataset_names:List[str],
                      data_path:str,
                      type:str='predict',
                      plot_type:str='samples',
                      plot_path:str=os.getcwd(),
                      show_plot:bool=True,
                      save_plot:bool=True)-> tuple[plt.figure, plt.axes]:
    
    """
    Obtain the time scaling plot.

    Args:
        model_names: The list of model names.
        dataset_names: The list of dataset names.
        data_path: The path to the datasets.
        type: The type of execution time, accepted values are: ['fit','predict','importances'] Defaults to 'predict'.
        plot_type: The type of plot, accepted values are ['samples','features']. Defaults to 'samples'.
        plot_path: The path where the plot will be saved. Defaults to os.getcwd().
        show_plot: A boolean indicating whether the plot should be displayed. Defaults to True.
        save_plot: A boolean indicating whether the plot should be saved. Defaults to True.

    Returns:
        The figure and axes objects used to create the plot.
    """

    assert type in ['predict','fit','importances'], "Type not valid. Accepted values: ['predict','fit','importances'] "
    assert plot_type in ['samples','features'], "Plot Type not valid. Accepted values: ['samples','features']"
    
    datasets=[Dataset(name,path=data_path) for name in dataset_names]

    if plot_type == "samples":
        sample_sizes=[data.shape[0] for data in datasets]
    elif plot_type == "features":
        sample_sizes=[data.shape[1] for data in datasets]

    fig, ax = plt.subplots()
    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = '#F2F2F2'
    plt.grid(alpha = 0.7)
    colors = ["tab:red","tab:blue","tab:orange","tab:green","tab:blue"]

    maxs=[]
    mins=[]
    for i,model in enumerate(model_names):
        median_times,five_times,ninefive_times=get_vals(model,dataset_names,type=type)
        maxs.append(np.max(median_times))
        mins.append(np.min(median_times))

        ax.plot(sample_sizes,median_times,alpha=0.85,c=colors[i],marker="o",label=model)
        ax.fill_between(sample_sizes,five_times,ninefive_times,alpha=0.1,color=colors[i])
    
    if plot_type == "samples":
        ax.set_yscale('log')
        ax.set_xscale("log")
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xticks(sample_sizes,sample_sizes,rotation=45,fontsize = 12)
        ax.set_yticks([1,10,25,100,250],fontsize = 14)
        
    ax.set_xlabel('Sample Size',fontsize = 20)
    ax.set_ylabel(f'{type} Time (s)',fontsize = 20)
    #plt.ylim(np.min(mins)-0.2*np.min(mins),np.max(maxs)+0.2*np.max(maxs))

    
    ax.legend()
    ax.grid(visible=True, alpha=0.5, which='major', color='gray', linestyle='-')
    
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    if save_plot:
        plt.savefig(f'{plot_path}/{current_time}_time_scaling_plot_{plot_type}_{type}.pdf',bbox_inches='tight')

    if show_plot:
        plt.show()
    
    return fig,ax

def plot_ablation(eta_list:List[float],
                  avg_prec:List[np.ndarray],
                  EIF_value:float,
                  dataset_name:str,
                  plot_path:str=os.getcwd(),
                  show_plot:bool=False,
                  save_plot:bool=True,
                  change_ylim:bool=False) -> tuple[plt.figure, plt.axes]:
    
    """
    Obtain the plot of the Average precision values against different values of the era parameter.

    Args:
        eta_list: The list of eta values.
        avg_prec: The list of average precision values.
        EIF_value: The average precision value of the EIF model.
        dataset_name: The dataset name.
        plot_path: The path where the plot will be saved. Defaults to os.getcwd().
        show_plot: A boolean indicating whether the plot should be displayed. Defaults to False.
        save_plot: A boolean indicating whether the plot should be saved. Defaults to True.
        change_ylim: A boolean indicating whether the y axis limits should be changed. Defaults to False.

    Returns:
        The figure and axes objects used to create the plot.
    """

    fig, ax = plt.subplots()
    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = '#F2F2F2'
    plt.grid(alpha = 0.7)
    colors = ["tab:red","tab:blue","tab:orange","tab:green","tab:blue"]


    median_values=[np.mean(x) for x in avg_prec]
    five_values=[np.percentile(x,5) for x in avg_prec]
    ninefive_values=[np.percentile(x,95) for x in avg_prec]

    ax.plot(eta_list,median_values,alpha=0.85,c=colors[0],marker="o",label="EIF+")
    ax.plot(eta_list,[EIF_value]*len(eta_list),alpha=0.85,c=colors[1],label="EIF")
    ax.fill_between(eta_list,five_values,ninefive_values,alpha=0.1,color=colors[0])

        
    ax.set_xlabel("Eta",fontsize = 20)
    ax.set_ylabel('Avg Prec',fontsize = 20)

    
    ax.grid(visible=True, alpha=0.5, which='major', color='gray', linestyle='-')
    
    if change_ylim:
        ax.set_ylim([0,1.1])
    else:
        ax.set_ylim([0,1])

    plt.legend()
    
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    if save_plot:
        plt.savefig(f'{plot_path}/{current_time}_EIF+_ablation_{dataset_name}.pdf',bbox_inches='tight')

    if show_plot:
        plt.show()
    
    return fig,ax

