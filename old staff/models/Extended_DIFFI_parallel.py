import sys
import os 
import pickle
import time 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm
from typing import List
import logging
from functools import partial
from multiprocessing import Pool, cpu_count


sys.path.append("./models")
from models.Extended_IF import ExtendedIF, ExtendedTree
import numpy as np
import pandas as pd


class Extended_DIFFI_tree(ExtendedTree):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.importances = []
        self.sum_normals = []

    @staticmethod
    def importance_worker(X, nodes, depth_based, left_son, right_son):
        """
        Compute the Importance Scores for each node along the Isolation Trees.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.

        Returns
        ----------
        Importances_list: np.array
                List with the Importances values for all the nodes in the Isolation Tree.
        Normal_vectors_list: np.array
                List of all the normal vectors used in the splitting hyperplane creation.
        """
        Importances_list = np.zeros((X.shape[0], X.shape[1]))
        Normal_vectors_list = np.zeros((X.shape[0], X.shape[1]))
        for i, x in enumerate(X):
            id = depth = 0
            while True:
                s = nodes[id]["point"]
                if s is None:
                    break
                n = nodes[id]["normal"]
                N = nodes[id]["numerosity"]
                old_id = id
                if x.dot(n) - s > 0:
                    side = "left_importance"
                    id = left_son[id]
                else:
                    side = "right_importance"
                    id = right_son[id]
                abs_n = np.abs(n)
                singular_importance = abs_n * (N / (nodes[id]["numerosity"] + 1))
                if depth_based == True:
                    singular_importance /= 1 + depth
                Importances_list[i] += singular_importance
                Normal_vectors_list[i] += abs_n
                nodes[old_id].setdefault(side, singular_importance)
                nodes[old_id].setdefault("depth", depth)
                depth += 1
        return Importances_list, Normal_vectors_list

    def make_importance(self, X, depth_based):
        """
        Compute the Importance Scores for each node along the Isolation Trees.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.

        Returns
        ----------
        Importances_list: np.array
                List with the Importances values for all the nodes in the Isolation Tree.
        Normal_vectors_list: np.array
                List of all the normal vectors used in the splitting hyperplane creation.
        """

        # multicore processing → this part is not parallelized here
        num_processes = 1  

        if num_processes > 1:
            partial_importance_worker = partial(
                self.importance_worker,
                nodes=self.nodes,
                depth_based=depth_based,
                left_son=self.left_son,
                right_son=self.right_son,
            )

            # split the input vector num_processes
            segment_size = len(X) // num_processes
            segments = [X[i : i + segment_size] for i in range(0, len(X), segment_size)]

            with Pool(processes=num_processes) as pool:
                results = pool.map(partial_importance_worker, segments)

                Importances_list = []
                Normal_vectors_list = []

                for result in results:
                    Importances_list.extend(result[0])
                    Normal_vectors_list.extend(result[1])
        else:
            Importances_list, Normal_vectors_list = self.importance_worker(
                X, self.nodes, depth_based, self.left_son, self.right_son
            )

        return np.array(Importances_list), np.array(Normal_vectors_list)


class Extended_DIFFI_parallel(ExtendedIF):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.sum_importances_matrix = None
        self.sum_normal_vectors_matrix = None
        self.plus = kwarg.get("plus")
        self.distribution = kwarg.get("distribution")
        self.eta = kwarg.get("eta")
        self.contamination = kwarg.get("contamination")
        self.num_processes_importances = 1
        self.num_processes_fit = 1
        self.num_processes_anomaly = 1
        self.distribution="normal_mean"

    def set_num_processes(
        self, num_processes_fit, num_processes_importances, num_processes_anomaly
    ):
        """
        Set the number of processes to be used in the parallel computation
        of the Global and Local Feature Importance.
        """
        self.num_processes_fit = num_processes_fit
        self.num_processes_importances = num_processes_importances
        self.num_processes_anomaly = num_processes_anomaly

    @property
    def contamination_(self):
        return self.contamination
    
    @property
    def distribution_(self):
        return self.distributionù
    
    @property
    def eta_(self):
        return self.eta

    @staticmethod
    def make_tree_worker(forest_segment: List[Extended_DIFFI_tree], X, subsample_size):
        # subsets = []
        for x in forest_segment:
            if not subsample_size or subsample_size > X.shape[0]:
                x.make_tree(X, 0, 0)
            else:
                indx = np.random.choice(X.shape[0], subsample_size, replace=False)
                X_sub = X[indx, :]
                x.make_tree(X_sub, 0, 0)
                # subsets.append(indx)
        return forest_segment

    def fit(self, X):
        """
        Fit the ExIFFI model.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset

        """
        if not self.dims:
            self.dims = X.shape[1]
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = np.inf

        if hasattr(self, "num_fit_calls"):
            self.num_fit_calls += 1
        else:
            # first call
            self.num_fit_calls = 0

        self.forest = [
            Extended_DIFFI_tree(
                dims=self.dims,
                min_sample=self.min_sample,
                max_depth=self.max_depth,
                plus=self.plus,
                distribution=self.distribution,
                eta=self.eta
            )
            for i in range(self.n_trees)
        ]

        if self.num_processes_fit > 1:
            # --- Parallel execution ---
            partial_make_tree_worker = partial(
                self.make_tree_worker, X=X, subsample_size=self.subsample_size
            )

            segment_size = max(1, len(self.forest) // self.num_processes_fit)

            segments = [
                self.forest[i : i + segment_size]
                for i in range(0, len(self.forest), segment_size)
            ]
            with Pool(processes=self.num_processes_fit) as pool:
                results = pool.map(partial_make_tree_worker, segments)

                self.forest = []
                for result in results:
                    self.forest.extend(result)

        else:
            # --- Serial execution ---
            self.subsets = []
            for x in self.forest:
                if not self.subsample_size or self.subsample_size > X.shape[0]:
                    x.make_tree(X, 0, 0)
                else:
                    indx = np.random.choice(
                        X.shape[0], self.subsample_size, replace=False
                    )
                    X_sub = X[indx, :]
                    x.make_tree(X_sub, 0, 0)
                    self.subsets.append(indx)

    @staticmethod
    def forest_worker(forest: List[Extended_DIFFI_tree], X, depth_based):
        """
        This takes a segment of the forest, which is a list of trees.
        Given a dataset X and the depth_based option, return a function that
        computes the sum of the importance scores and the sum of the normal vectors.
        """
        # forest, X, depth_based = args

        partial_sum_importances_matrix = np.zeros_like(X, dtype="float64")
        partial_sum_normal_vectors_matrix = np.zeros_like(X, dtype="float64")

        for tree in forest:
            importances_matrix, normal_vectors_matrix = tree.make_importance(
                X, depth_based
            )
            partial_sum_importances_matrix += importances_matrix
            partial_sum_normal_vectors_matrix += normal_vectors_matrix

        return partial_sum_importances_matrix, partial_sum_normal_vectors_matrix

    def Importances(self, X, calculate, overwrite, depth_based):
        """
        Obtain the sum of the Importance scores computed along all the Isolation Trees, with the make_importance
        function.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.
        calculate: bool
                If calculate is True the Importances Sum Matrix and the Normal Vectors Sum Matrix are initialized to 0
        overwrite: bool
                Boolean variable used to decide weather to overwrite evrytime the value inserted in sum_importances_matrix and
                in sum_normal_vectors_matrix.

        Returns
        ----------
        sum_importances_matrix: np.array
                2-dimensional array containing,for each sample, the sum of the importance scores obtained by the nodes in which
                it was included.
        sum_normal_vectors_matrix: np.array
                2-dimensional array containing,for each sample, the sum of the normal vectors used to create the
                splitting hyperplances of the nodes in which it was included.

        """
        if (self.sum_importances_matrix is None) or calculate:
            sum_importances_matrix = np.zeros_like(X, dtype="float64")
            sum_normal_vectors_matrix = np.zeros_like(X, dtype="float64")

            # multicore processing
            # split the input vector into segments
            # from this: [tree0, tree1, tree2, tree3, tree4]
            # to this: [  [tree0, tree1],   [tree2, tree3], [tree4]]]

            segment_size = max(1, len(self.forest) // self.num_processes_importances)

            segments = [
                self.forest[i : i + segment_size]
                for i in range(0, len(self.forest), segment_size)
            ]


            if self.num_processes_importances > 1:
                with Pool(processes=self.num_processes_importances) as pool:
                    forest_worker_partial = partial(
                        self.forest_worker, X=X, depth_based=depth_based
                    )

                    # the result list of tuples which are the outputs of the make_importance function
                    results = pool.map(forest_worker_partial, segments)

                    # results = [ (part_sum_mat0, part_sum_norm0), # worker 0
                    #             (part_sum_mat1, part_sum_norm1), # worker 1
                    #             ...
                    #           ]

                    for result in results:
                        sum_importances_matrix += result[0]
                        sum_normal_vectors_matrix += result[1]

                    # the division can be done at the end
                    sum_importances_matrix /= self.n_trees
                    sum_normal_vectors_matrix /= self.n_trees
            else:
                sum_importances_matrix, sum_normal_vectors_matrix = self.forest_worker(
                    self.forest, X, depth_based
                )

            if overwrite:
                self.sum_importances_matrix = sum_importances_matrix / self.n_trees
                self.sum_normal_vectors_matrix = (
                    sum_normal_vectors_matrix / self.n_trees
                )

            return sum_importances_matrix, sum_normal_vectors_matrix
        else:
            return self.sum_importances_matrix, self.sum_normal_vectors_matrix

    def Global_importance(self, X, calculate, overwrite, depth_based=False):
        """
        Compute the Global Feature Importance vector for a set of input samples
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        calculate: bool
                Used to call the Importances function
        overwrite: bool
                Used to call the Importances function.
         depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.
                By default the value of depth_based is False.

        Returns
        ----------
        Global_Importance: np.array
        Array containig a Global Feature Importance Score for each feature in the dataset.

        """

        anomaly_scores = self.Anomaly_Score(X)
        ind = np.argpartition(anomaly_scores, -int(self.contamination * len(X)))[-int(self.contamination * len(X)) :]
        importances_matrix, normal_vectors_matrix = self.Importances(
            X, calculate, overwrite, depth_based
        )

        Outliers_mean_importance_vector = np.mean(importances_matrix[ind], axis=0)
        Inliers_mean_Importance_vector = np.mean(
            importances_matrix[np.delete(range(len(importances_matrix)), ind)], axis=0
        )

        Outliers_mean_normal_vector = np.mean(normal_vectors_matrix[ind], axis=0)
        Inliers_mean_normal_vector = np.mean(
            normal_vectors_matrix[np.delete(range(len(importances_matrix)), ind)],
            axis=0,
        )

        return (Outliers_mean_importance_vector / Outliers_mean_normal_vector) / (
            Inliers_mean_Importance_vector / Inliers_mean_normal_vector
        ) - 1

    def Local_importances(self, X, calculate, overwrite, depth_based=False):
        """
        Compute the Local Feature Importance vector for a set of input samples
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        calculate: bool
                Used to call the Importances function
        overwrite: bool
                Used to call the Importances function.
         depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.
                By default the value of depth_based is False.

        Returns
        ----------
        Local_Importance: np.array
        Array containig a Local Feature Importance Score for each feature in the dataset.

        """
        importances_matrix, normal_vectors_matrix = self.Importances(
            X, calculate, overwrite, depth_based
        )
        return importances_matrix / normal_vectors_matrix
    
    def compute_local_importances(self,
                                  X: np.array,
                                  name: str,
                                  calculate: bool,
                                  overwrite: bool,
                                  depth_based: bool = False,
                                  pwd_imp_score: str = os.getcwd(), 
                                  pwd_plt_data: str = os.getcwd(),) -> tuple[np.array,dict,str,str]:
        """
        Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
        functions. 
        
        Parameters
        ----------
        X: Input dataset,np.array of shape (n_samples,n_features)
        name: Dataset's name   
        calculate: Parameter of the Local_importances function
        overwrite: Parameter of the Local_importances function
        depth_based: Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance in Local_importances
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
        fi=self.Local_importances(X,calculate,overwrite,depth_based)

        # Handle the case in which there are some np.nan or np.inf values in the fi array
        if np.isnan(fi).any() or np.isinf(fi).any():
            #Substitute the np.nan values with 0 and the np.inf values with the maximum value of the fi array plus 1. 
            fi=np.nan_to_num(fi,nan=0,posinf=np.nanmax(fi[np.isfinite(fi)])+1)
        
        # Save the Importance Scores in a npz file (more efficient than pkl files if we are using Python objects)
        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        filename = current_time + '_imp_scores_parallel_' + name + '.npz'
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
        filename_plt = current_time + '_plt_data_parallel_' + name + '.pkl'
        path_plt_data = pwd_plt_data + '/' + filename_plt
        with open(path_plt_data, 'wb') as fl:
            pickle.dump(plt_data,fl)
        

        return fi,plt_data,path_fi,path_plt_data
    
    def compute_global_importances(self,
                                   X: np.array, 
                                   n_runs:int, 
                                   name: str,
                                   calculate: bool,
                                   overwrite: bool,
                                   depth_based: bool = False,
                                   pwd_imp_score: str = os.getcwd(),
                                   pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
            """
            Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
            functions. 
            
            Parameters
            ----------
            X: Input Dataset,np.array of shape (n_samples,n_features)
            n_runs: Number of runs to perform in order to compute the Global Feature Importance Scores.
            name: Dataset's name   
            calculate: Parameter of the Global_importance function
            overwrite: Parameter of the Global_importance function
            depth_based: Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance in Global_importance
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
                    self.fit(X)
                    fi[i,:]=self.Global_importance(X,calculate,overwrite,depth_based)

            # Handle the case in which there are some np.nan or np.inf values in the fi array
            if np.isnan(fi).any() or np.isinf(fi).any():
                    #Substitute the np.nan values with 0 and the np.inf values with the maximum value of the fi array plus 1. 
                    fi=np.nan_to_num(fi,nan=0,posinf=np.nanmax(fi[np.isfinite(fi)])+1)

            # Save the Importance Scores in a npz file (more efficient than pkl files if we are using Python objects)
            t = time.localtime()
            current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
            filename = current_time + '_imp_scores_parallel_' + name + '.npz'
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
            filename_plt = current_time + '_plt_data_parallel_' + name + '.pkl'
            path_plt_data = pwd_plt_data + '/' + filename_plt
            with open(path_plt_data, 'wb') as fl:
                    pickle.dump(plt_data,fl)
            

            return fi,plt_data,path_fi,path_plt_data
    
    def bar_plot(self,
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
        fig,ax : plt.figure  and plt.axes objects used to create the plot 
        bars: pd.DataFrame containing the percentage count of the features in the first f positions of the Bar Plot.    
        """

        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        name_file = current_time + '_GFI_Bar_plot_parallel_' + name 

        if is_local:
            name_file = current_time + '_LFI_Bar_plot_parallel_' + name
        
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


    def score_plot(self,
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
    

    def importance_map(self,
                       name: str, 
                       X_train: np.array,
                       y_train: np.array,
                       resolution: int,
                       pwd: str =os.getcwd(),
                       save: bool =True,
                       m: bool =None,
                       factor: int =3, 
                       feats_plot: tuple =(0,1),
                       col_names=None,
                       ax=None,
                       labels: bool=True,
                       show_plot: bool =True):
        """
        Produce the Local Feature Importance Scoremap.   
        
        Parameters
        ----------
        name: Dataset's name
        X_train: Training Set 
        y_train: Dataset training labels
        resolution: Scoremap resolution 
        pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
        save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
        m: Boolean variable regulating the plt.pcolor advanced settings. By defualt the value of m is set to None.
        factor: Integer factor used to define the minimum and maximum value of the points used to create the scoremap. By default the value of f is set to 3.
        feats_plot: This tuple contains the indexes of the pair features to compare in the Scoremap. By default the value of feats_plot
        is set to (0,1). Do not use in case we pass the col_names parameter.
        col_names: List with the names of the features of the input dataset, by default None.
        two features will be compared. 
        ax: plt.axes object used to create the plot. By default ax is set to None.
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
        importance_matrix = self.Local_importances(mean, True, False)
        
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
	
    def importance_map_col_names(self,
                                 name: str,
                                 X:pd.DataFrame,
                                 X_train: np.array,
                                 y_train: np.array,
                                 resolution: int,
                                 pwd: str =os.getcwd(),
                                 save: bool =True,
                                 m: bool =None,
                                 factor: int =3, 
                                 col_names=None,
                                 ax=None,
                                 labels: bool=True,
                                 show_plot: bool =True):
          
            """Stub method of plot_importance_map used to give the user the possibility of specifying the names of the features to compare in the Scoremap.
            
            Parameters
            ----------
            name: Dataset's name
            X: Input dataset as a pd.DataFrame
            X_train: Training Set
            y_train: Dataset training labels
            resolution: Scoremap resolution
            pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
            save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. By default save is set to True.
            m: Boolean variable regulating the plt.pcolor advanced settings. By defualt the value of m is set to None.
            factor: Integer factor used to define the minimum and maximum value of the points used to create the scoremap. By default the value of f is set to 3.
            col_names: List with the names of the two features that will be compares, by default None.
            ax: plt.axes object used to create the plot. By default ax is set to None.
            labels: Boolean variable used to decide weather to include the x and y label name in the plot.
            """
            
            feats_plot=[X.columns.get_loc(col_names[0]),X.columns.get_loc(col_names[1])]           
            col_names=list(X.columns)

            return self.importance_map(name,X_train,y_train,resolution,pwd,save,m,factor,feats_plot,col_names,ax,labels,show_plot)
	 

    def complete_scoremap(self,
                          name:str,
                          dim:int,
                          X: pd.DataFrame,
                          y: np.array,
                          pwd:str =os.getcwd(),
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
        pwd: Directory where the plot will be saved as a PDF file. By default the value of pwd is set to the current working directory.
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
                    _,_=self.importance_map(name,X, y, 50, pwd, feats_plot = (features[0],features[1]), ax=ax[i,j],save=False,labels=False)
                    if half:
                        continue
                    _,_=self.importance_map(name,X, y, 50, pwd, feats_plot = (features[1],features[0]), ax=ax[j,i],save=False,labels=False)

        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        filename = current_time + '_Local_Importance_Scoremap_parallel_' + name + '_complete'

        if save:
            plt.savefig(pwd+'/{}.pdf'.format(filename),bbox_inches='tight')

        if show_plot:
            plt.show()
    
        return fig,ax

