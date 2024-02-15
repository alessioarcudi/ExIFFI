import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm
from matplotlib.pyplot import *
import pickle
from tqdm import tqdm
import sys;sys.path.append('../')

from models.interpretability_module import local_diffi
from models.forests import IsolationForest
from model_reboot.EIF_reboot import *


"""
Insert here the newest version of the plot functions used in the ExIFFI model and adapted to work with the ExIFFI implementation contained in 
EIF_reboot.py. 
"""

def compute_local_importances(X: np.array,
                              name: str,
                              model=ExtendedIsolationForest(),
                              depth_based: bool = False,
                              pwd_imp_score: str = os.getcwd(), 
                              pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
        """
        Computhe the Local Feature Importance Scores for the input dataset X. Save them into a npz file and save the plt_data dictionary into a pkl file. 
        
        Parameters
        ----------
        X: Input dataset,np.array of shape (n_samples,n_features)
        name: Dataset's name  
        model: Instance of the model on which we want to compute the Local Feature Importance Scores,  by default the default version of ExtendedIsolationForest
        depth_based: Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance in Local_importances, by default False
        pwd_imp_score: Directory where the Importance Scores results will be saved as pkl files, by default the current working directory
        pwd_plt_data: Directory where the plot data results will be saved as pkl files, by default the current working directory        
    
        Returns
        ----------
        imps: array of shape (n_samples,n_features) containing the local Feature Importance values for the samples of the input dataset X. 
        The array is also locally saved in a pkl file for the sake of reproducibility.
        plt_data: Dictionary containig the average Importance Scores values, the feature order and the standard deviations on the Importance Scores. 
        The dictionary is also locally saved in a pkl file for the sake of reproducibility.
        path_fi: Path of the pkl file containing the Importance Scores.
        path_plt_data: Path of the pkl file containing the plt data.    
        """

        name='LFI_'+name
        fi=model.local_importances(X)

        # Handle the case in which there are some np.nan or np.inf values in the fi array
        if np.isnan(fi).any() or np.isinf(fi).any():
            #Substitute the np.nan values with 0 and the np.inf values with the maximum value of the fi array plus 1. 
            fi=np.nan_to_num(fi,nan=0,posinf=np.nanmax(fi[np.isfinite(fi)])+1)
        
        # Save the Importance Scores in a npz file (more efficient than pkl files if we are using Python objects)
        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        filename = current_time + '_imp_scores_reboot_' + name + '.npz'
        path_fi = pwd_imp_score  + '/' + filename
        np.savez(path_fi,fi=fi)

        """ 
        Take the mean feature importance scores over the different runs for the Feature Importance Plot
        and put it in decreasing order of importance.
        To remove the possible np.nan or np.inf values from the mean computation use assign np.nan to the np.inf values 
        and then ignore the np.nan values using np.nanmean
        """

        fi[fi==np.inf]=np.nan
        mean_imp=np.nanmean(fi,axis=0)
        std_imp=np.nanstd(fi,axis=0)
        mean_imp_val=np.sort(mean_imp)
        feat_order=mean_imp.argsort()

        plt_data={'Importances': mean_imp_val,
                'feat_order': feat_order,
                'std': std_imp[mean_imp.argsort()]}
        
        # Save the plt_data dictionary in a pkl file
        filename_plt = current_time + '_plt_data_reboot_' + name + '.pkl'
        path_plt_data = pwd_plt_data + '/' + filename_plt
        with open(path_plt_data, 'wb') as fl:
            pickle.dump(plt_data,fl)
        

        return fi,plt_data,path_fi,path_plt_data
    
def compute_global_importances(X: np.array, 
                                n_runs: int, 
                                name: str,
                                model=ExtendedIsolationForest(),
                                p:float=0.1,
                                depth_based: bool = False,
                                pwd_imp_score: str = os.getcwd(),
                                pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
        """
        Computhe the Global Feature Importance Scores for the input dataset X. Save them into a npz file and save the plt_data dictionary into a pkl file.
        
        Parameters
        ----------
        X: Input Dataset,np.array of shape (n_samples,n_features)
        n_runs: Number of runs to perform in order to compute the Global Feature Importance Scores.
        name: Dataset's name   
        model: Instance of the model on which we want to compute the Local Feature Importance Scores,  by default the default version of ExtendedIsolationForest
        p: Contamination factor of the dataset, by default 0.1
        depth_based: Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance in Global_importance, by default False
        pwd_imp_score: Directory where the Importance Scores results will be saved as pkl files, by default the current working directory
        pwd_plt_data: Directory where the plot data results will be saved as pkl files, by default the current working directory        
                                
        Returns
        ----------
        imps: array of shape (n_samples,n_features) containing the local Feature Importance values for the samples of the input dataset X. 
        The array is also locally saved in a pkl file for the sake of reproducibility.
        plt_data: Dictionary containing the average Importance Scores values, the feature order and the standard deviations on the Importance Scores. 
        The dictionary is also locally saved in a pkl file for the sake of reproducibility.
        path_fi: Path of the pkl file containing the Importance Scores
        path_plt_data: Path of the pkl file containing the plt data    
        """

        name='GFI_'+name
        fi=np.zeros(shape=(n_runs,X.shape[1]))
        for i in range(n_runs):
                model.fit(X)
                fi[i,:]=model.global_importances(X,p)

        # Handle the case in which there are some np.nan or np.inf values in the fi array
        if np.isnan(fi).any() or np.isinf(fi).any():
                #Substitute the np.nan values with 0 and the np.inf values with the maximum value of the fi array plus 1. 
                fi=np.nan_to_num(fi,nan=0,posinf=np.nanmax(fi[np.isfinite(fi)])+1)

        # Save the Importance Scores in a npz file (more efficient than pkl files if we are using Python objects)
        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        filename = current_time + '_imp_scores_reboot_' + name + '.npz'
        path_fi = pwd_imp_score  + '/' + filename
        np.savez(path_fi,fi=fi)
                

        fi[fi==np.inf]=np.nan
        mean_imp=np.nanmean(fi,axis=0)
        std_imp=np.nanstd(fi,axis=0)
        mean_imp_val=np.sort(mean_imp)
        feat_order=mean_imp.argsort()

        plt_data={'Importances': mean_imp_val,
                        'feat_order': feat_order,
                        'std': std_imp[mean_imp.argsort()]}
        
        # Save the plt_data dictionary in a pkl file
        filename_plt = current_time + '_plt_data_reboot_' + name + '.pkl'
        path_plt_data = pwd_plt_data + '/' + filename_plt
        with open(path_plt_data, 'wb') as fl:
                pickle.dump(plt_data,fl)
        

        return fi,plt_data,path_fi,path_plt_data

def bar_plot(
          imps_path: str,
          name: str,
          pwd: str =os.getcwd(),
          f: int = 6,
          col_names = None,
          is_local: bool=False,
          save: bool =True,
          show_plot: bool =True):
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
        fig,ax : plt.figure and plt.axes objects used to create the plot 
        bars: pd.DataFrame containing the percentage count of the features in the first f positions of the Bar Plot.    
        """

        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        name_file = current_time + '_GFI_Bar_plot_reboot_' + name 

        if is_local:
            name_file = current_time + '_LFI_Bar_plot_reboot_' + name
        
        #Load the imps array from the pkl file contained in imps_path -> the imps_path is returned from the 
        #compute_local_importances or compute_global_importances functions so we have it for free 
        importances=np.load(imps_path)['fi']

        number_colours = 20
        color = plt.cm.get_cmap('tab20',number_colours).colors
        patterns=[None,'!','@','#','$','^','&','*','°','(',')','-','_','+','=','[',']','{','}',
        '|',';',':','\l',',','.','<','>','/','?','`','~','\\','!!','@@','##','$$','^^','&&','**','°°','((']
        importances_matrix = np.array([np.array(pd.Series(x).sort_values(ascending = False).index).T for x in importances])
        dim=importances.shape[1]
        dim=int(dim)
        bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
        bars = pd.DataFrame(bars)

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
            plt.savefig(pwd + '/{}.pdf'.format(name_file), bbox_inches='tight')

        if show_plot:
            plt.show()

        return fig, ax, bars

def score_plot(
               plt_data_path: str,
               name: str,
               pwd: str =os.getcwd(),
               col_names=None,
               is_local: bool =False,
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
        with open(plt_data_path, 'rb') as f:
            plt_data = pickle.load(f)

        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        name_file = current_time + '_GFI_Score_plot_parallel_' + name 

        if is_local:
            name_file = current_time + '_LFI_Score_plot_parallel_' + name

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

        if col_names is not None:
            ax1.set_yticks(range(dim))
            idx=list(feat_imp['Feature'])
            yticks=[col_names[i] for i in idx]
            ax1.set_yticklabels(yticks)

        if save:
            plt.savefig(pwd+'/{}.pdf'.format(name_file),bbox_inches='tight')

        if show_plot:
            plt.show()
            
        return ax1,ax2

def importance_map(
                   name: str, 
                   X_train: np.array,
                   y_train: np.array,
                   model=ExtendedIsolationForest(),
                   iforest=IsolationForest(),
                   resolution: int = 30,
                   pwd: str = os.getcwd(),
                   save: bool = True,
                   m: bool = None,
                   factor: int = 3, 
                   feats_plot: tuple = (0,1),
                   col_names: List[str] = None,
                   ax=None,
                   isdiffi: bool = False,
                   labels: bool = True,
                   show_plot: bool = True):
        """
        Produce the Local Feature Importance Scoremap.   
        
        Parameters
        ----------
        name: Dataset's name
        X_train: Training Set 
        y_train: Dataset training labels
        model: Instance of the model on which we want to compute the Local Feature Importance Scores, by default the default version of ExtendedIsolationForest
        iforest: Isolation Forest model to use in case isdiffi is set to True, by default the default version of IsolationForest
        resolution: Scoremap resolution, by default 30
        pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
        save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
        m: Boolean variable regulating the plt.pcolor advanced settings. By defualt the value of m is set to None.
        factor: Integer factor used to define the minimum and maximum value of the points used to create the scoremap. By default the value of f is set to 3.
        feats_plot: This tuple contains the indexes of the pair of features to compare in the Scoremap. By default the value of feats_plot
        is set to (0,1). Do not use in case we pass the col_names parameter.
        col_names: List with the names of the features of the input dataset, by default None.
        two features will be compared. 
        ax: plt.axes object used to create the plot. By default ax is set to None.
        isdiffi: Boolean variable used to decide weather to use the Diffi algorithm to compute the Local Feature Importance Scores or not. By default isdiffi is set to False.
        labels: Boolean variable used to decide weather to include the x and y label name in the plot.
        When calling the plot_importance_map function inside plot_complete_scoremap this parameter will be set to False 
        show_plot: Boolean variable used to decide whether to show the plot on the screen or not. By default, show_plot is set to True.
                    
        Returns
        ----------
        fig,ax : plt.figure  and plt.axes objects used to create the plot 
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
                iforest.max_samples = len(X_train)
                for i in range(importance_matrix.shape[0]):
                        importance_matrix[i] = local_diffi(iforest, mean[i])[0]
        else:
                importance_matrix = model.local_importances(mean)
        
        sign = np.sign(importance_matrix[:,feats_plot[0]]-importance_matrix[:,feats_plot[1]])
        Score = sign*((sign>0)*importance_matrix[:,feats_plot[0]]+(sign<0)*importance_matrix[:,feats_plot[1]])
        x = X_train[:,feats_plot[0]].squeeze()
        y = X_train[:,feats_plot[1]].squeeze()
        
        Score = Score.reshape(xx.shape)

        # Create a new pyplot object if plt is not provided
        if ax is None:
            fig, ax = plt.subplots()
        
        if m is not None:
            cp = ax.pcolor(xx, yy, Score, cmap=cm.RdBu, vmin=-m, vmax=m, shading='nearest')
        else:
            cp = ax.pcolor(xx, yy, Score, cmap=cm.RdBu, shading='nearest', norm=colors.CenteredNorm())
        
        ax.contour(xx, yy, (importance_matrix[:, feats_plot[0]] + importance_matrix[:, feats_plot[1]]).reshape(xx.shape), levels=7, cmap=cm.Greys, alpha=0.7)

        try:
            ax.scatter(x[y_train == 0], y[y_train == 0], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
            ax.scatter(x[y_train == 1], y[y_train == 1], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")
        except IndexError:
            print('Handling the IndexError Exception...')
            ax.scatter(x[(y_train == 0)[:, 0]], y[(y_train == 0)[:, 0]], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
            ax.scatter(x[(y_train == 1)[:, 0]], y[(y_train == 1)[:, 0]], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")
        
        if (labels) and (col_names is not None):
            ax.set_xlabel(col_names[feats_plot[0]],fontsize=20)
            ax.set_ylabel(col_names[feats_plot[1]],fontsize=20)
        elif (labels) and (col_names is None):
            ax.set_xlabel(f'Feature {feats_plot[0]}',fontsize=20)
            ax.set_ylabel(f'Feature {feats_plot[1]}',fontsize=20)
        
        ax.legend()

        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        filename = current_time + '_Local_Importance_Scoremap_parallel_' + name

        if save:
            plt.savefig(pwd + '/{}.pdf'.format(filename), bbox_inches='tight')
        else: 
            fig,ax=None,None

        if show_plot:
            plt.show()

        return fig, ax

def importance_map_col_names(
                            name: str,
                            X:pd.DataFrame,
                            X_train: np.array,
                            y_train: np.array,
                            model=ExtendedIsolationForest(),
                            iforest=IsolationForest(),
                            resolution: int = 30,
                            pwd: str =os.getcwd(),
                            save: bool =True,
                            m: bool =None,
                            factor: int =3, 
                            col_names=None,
                            ax=None,
                            isdiffi: bool = False,
                            labels: bool=True,
                            show_plot: bool =True):
          
            """Stub method of plot_importance_map used to give the user the possibility of specifying the names of the features to compare in the Scoremap.
            
            Parameters
            ----------
            name: Dataset's name
            X: Input dataset as a pd.DataFrame
            X_train: Training Set
            y_train: Dataset training labels
            model: Instance of the model on which we want to compute the Local Feature Importance Scores, by default the default version of ExtendedIsolationForest
            iforest: Isolation Forest model to use in case isdiffi is set to True, by default the default version of IsolationForest
            resolution: Scoremap resolution, by default 30
            pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
            save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
            m: Boolean variable regulating the plt.pcolor advanced settings. By defualt the value of m is set to None.
            factor: Integer factor used to define the minimum and maximum value of the points used to create the scoremap. By default the value of f is set to 3.
            col_names: List with the names of the two features that will be compares, by default None.
            ax: plt.axes object used to create the plot. By default ax is set to None.
            isdiffi: Boolean variable used to decide weather to use the Diffi algorithm to compute the Local Feature Importance Scores or not. By default isdiffi is set to False.
            labels: Boolean variable used to decide weather to include the x and y label name in the plot.
            """
            
            feats_plot=[X.columns.get_loc(col_names[0]),X.columns.get_loc(col_names[1])]           
            col_names=list(X.columns)

            return importance_map(name,X_train,y_train,model,iforest,resolution,pwd,save,m,factor,feats_plot,col_names,ax,isdiffi,labels,show_plot)

def complete_scoremap(
                     name:str,
                     dim:int,
                     X: pd.DataFrame,
                     y: np.array,
                     model=ExtendedIsolationForest(),
                     iforest=IsolationForest(),
                     pwd:str =os.getcwd(),
                     isdiffi: bool = False,
                     half: bool = False,
                     save: bool =True,
                     show_plot: bool =True):

        """Produce the Complete Local Feature Importance Scoremap: a Scoremap for each pair of features in the input dataset.   
        
        Parameters
        ----------
        name: Dataset's name
        dim: Number of input features in the dataset
        X: Input dataset 
        y: Dataset labels
        model: Instance of the model on which we want to compute the Local Feature Importance Scores, by default the default version of ExtendedIsolationForest
        iforest: Isolation Forest model to use in case isdiffi is set to True, by default the default version of IsolationForest
        pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
        isdiffi: Boolean variable used to decide weather to use the Diffi algorithm to compute the Local Feature Importance Scores or not. By default isdiffi is set to False.
        half: Boolean parameter to decide weather to plot all the possible scoremaps or only half of them, by default False
        save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
        show_plot: Boolean variable used to decide whether to show the plot on the screen or not. By default, show_plot is set to True.
        
        Returns
        ----------
        fig,ax : plt.figure  and plt.axes objects used to create the plot  
        """
            
        fig, ax = plt.subplots(dim, dim, figsize=(50, 50))
        for i in range(dim):
            for j in range(i+1,dim):
                    features = [i,j] 
                    _,_=importance_map(name,X,y,model,iforest,50,pwd,feats_plot=(features[0],features[1]),ax=ax[i,j],isdiffi=isdiffi,save=False,labels=False)
                    if half:
                        continue
                    _,_=importance_map(name,X,y,model,iforest,50,pwd,feats_plot=(features[1],features[0]),ax=ax[j,i],isdiffi=isdiffi,save=False,labels=False)

        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        filename = current_time + '_Local_Importance_Scoremap_parallel_' + name + '_complete'

        if save:
            plt.savefig(pwd+'/{}.pdf'.format(filename),bbox_inches='tight')

        if show_plot:
            plt.show()
    
        return fig,ax

"""
For the moment let's leave the old code commented 
"""

"""
def plt_importances_bars(importances, name, pwd, dim, f = 6):

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

    
    if 'GFI_' not in name:
        name='LFI_'+name
    number_colours = 20
    color = plt.cm.get_cmap('tab20',number_colours).colors
    patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    importances_matrix = np.array([np.array(pd.Series(x).sort_values(ascending = False).index).T for x in importances])
    bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
    bars = pd.DataFrame(bars)
    print(bars)

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
    

    path = pwd + '\\results\\davide\\Importance_Scores\\Imp Score\\imp_score_GFI_' + name + '.pkl'
    with open(path, 'rb') as fl:
        imps = pickle.load(fl)

    path = pwd + '\\results\\davide\\Importance_Scores\\Plt Data\\plt_data_GFI_' + name + '.pkl'
    with open(path, 'rb') as fl:
        plt_data = pickle.load(fl)

    plt_importances_bars(imps,name,pwd,dim)
    plt_feat_bar_plot(plt_data,name,pwd)
    


def plot_importance_map(name,model,X_train,y_train,resolution,pwd,save=True,m=None,factor=3,feats_plot=(0,1),plt=plt,isdiffi=False):
    
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

"""

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