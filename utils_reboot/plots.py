from __future__ import annotations

import os
import time
from typing import Type, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from utils_reboot.datasets import Dataset


def bar_plot(dataset: Type[Dataset], 
            global_importances_file: str, 
            plot_path: str = os.getcwd(), 
            f: int = 6, 
            col_names: Optional[list] = None,
            save = True, 
            show_plot = True) -> tuple[plt.figure, plt.axes, pd.DataFrame]:
    """
    Obtain the Global Importance Bar Plot given the Importance Scores values computed in the compute_local_importance or compute_global_importance functions. 
    
    Parameters
    ----------
    imps_path: Path of the pkl file containing the array of shape (n_samples,n_features) with the LFI/GFI Scores for the input dataset.
    Obtained from the compute_local_importance or compute_global_importance functions.   
    name: Dataset's name 
    pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.    
    f: Number of vertical bars to include in the Bar Plot. By default f is set to 6.
    col_names: List with the names of the features of the input dataset, by default None. 
    is_local: Boolean variable used to specify weather we are plotting the Global or Local Feature Importance in order to set the file name.
    If is_local is True the result will be the LFI Score Plot (based on the LFI scores of the input samples), otherwise the result is the GFI 
    Score Plot (based on the GFI scores obtained in the different n_runs execution of the model). By default is_local is set to False.  
    save: Boolean variable used to decide weather to save the Bar Plot locally as a PDF or not. BY default save is set to True. 
    show_plot: Boolean variable used to decide whether to show the plot on the screen or not. By default, show_plot is set to True.

    Returns
    ----------
    fig,ax : plt.figure  and plt.axes objects used to create the plot 
    bars: pd.DataFrame containing the percentage count of the features in the first f positions of the Bar Plot.    
    """

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    name_file = current_time + '_GFI_Bar_plot_parallel_' + dataset.name 
    
    #Load the imps array from the pkl file contained in imps_path -> the imps_path is returned from the 
    #compute_local_importances or compute_global_importances functions so we have it for free 
    try:
        importances=np.load(global_importances_file)['fi']
    except:
        raise Exception("The file path is not valid")

    number_colours = 20
    color = plt.cm.get_cmap('tab20',number_colours).colors
    patterns=[None,'!','@','#','$','^','&','*','°','(',')','-','_','+','=','[',']','{','}',
    '|',';',':','\l',',','.','<','>','/','?','`','~','\\','!!','@@','##','$$','^^','&&','**','°°','((']
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

    if save:
        plt.savefig(plot_path + '/{}.pdf'.format(name_file), bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig, ax, bars
    
    

def score_plot(dataset: Type[Dataset],
               global_importances_file: str,
               plot_path: str = os.getcwd(),
               col_names=None,
               save: bool =True,
               show_plot: bool =True):
    """
    Obtain the Global Feature Importance Score Plot exploiting the information obtained from the compute_local_importance or compute_global_importance functions. 
    
    Parameters
    ----------
    plt_data_path: Dictionary generated from the compute_local_importance or compute_global_importance functions 
    with the necessary information to create the Score Plot.
    name: Dataset's name
    pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory. 
    col_names: List with the names of the features of the input dataset, by default None.  
    is_local: Boolean variable used to specify weather we are plotting the Global or Local Feature Importance in order to set the file name.
    If is_local is True the result will be the LFI Score Plot (based on the LFI scores of the input samples), otherwise the result is the GFI 
    Score Plot (based on the GFI scores obtained in the different n_runs execution of the model). By default is_local is set to False. 
    save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
    show_plot: Boolean variable used to decide whether to show the plot on the screen or not. By default, show_plot is set to True.
                
    Returns
    ----------
    ax1,ax2: The two plt.axes objects used to create the plot.  
    """
    #Load the plt_data dictionary from the pkl file contained in plt_data_path -> the plt_data_path is returned from the 
    #compute_local_importances or compute_global_importances functions so we have it for free 
    with open(global_importances_file, 'rb') as f:
        plt_data = pickle.load(f)

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    name_file = current_time + '_GFI_Score_plot_' + dataset.name 

    patterns=[None,'!','@','#','$','^','&','*','°','(',')','-','_','+','=','[',']','{','}',
    '|',';',':','\l',',','.','<','>','/','?','`','~','\\','!!','@@','##','$$','^^','&&','**','°°','((']
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

    if save:
        plt.savefig(plot_path+'/{}.pdf'.format(name_file),bbox_inches='tight')

    if show_plot:
        plt.show()
        
    return ax1,ax2