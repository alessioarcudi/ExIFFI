# General Scheme 

> [!note] Color Code
> - <span style="color:red;"> Red  → Davide</span>
> - <span style="color:green;"> Green  → Alessio</span>
> - <span style="color:yellow;"> Yellow  → Davide and Alessio</span>
>  

- [x] [Possible solution to the Zoom Share Screen issue](https://technoracle.com/how-to-fix-zoom-screen-sharing-on-ubuntu-22-04-quickly/)

- [x] Try to organize better the GitHub repository so that also Alessio can work on that 
	- [x] Create Obsidian Vault to and add it to the Git Repository

> [!done] Git Repository
> GitHub Repository re organized and divided in different branches: 
> - `main` → Davide added things to it → from now on it will be better to start working on a separate branch for Davide stuff. The `main` will put together all the stuff done in the other branches once they are completed
> - `model_reboot` → Alessio used this branch to optimize the code with `numba`
> - `plot` → Branch to work on the functions to produce the plots 

- [x] Organize better the code to optimize it.
	- [x] Written `make_importance` function in C with `OpenMP` → up to 130 times faster  
	- [x] <span style="color:red;">Add the new features inserted in the plot functions for inserting DIFFI in PyOD</span>
	- [x]  <span style="color:green;">Reboot and review old code</span> 

> [!done] Model Reboot Code
>  Alessio has optimized and re organized the code in the `model_reboot` branch (now merged into `main`). Now we have: 
>  - `datasets.py` (inside folder `utils_reboot`) → `@dataclass` to create objects representing the datasets. We can use this class to `load` data, `drop_duplicates`, `downsample`, `partition_data` into training and test set and we can also obtain the `print_dataset_resume`. 
>  - `EIF_reboot.py` (inside folder `model_reboot`) → This is the model we will use in the paper. This code essentially puts together the code we had in `forests.py` and in `Extended_DIFFI_parallel.py`. 
>  - `trial.ipynb` → Notebook used to try out the `EIF_reboot.py` code 

> [!missing] Things to add
> - Move `datasets.py` from folder `utils_reboot` to folder `src`
> - Move `EIF_reboot.py` from folder `model_reboot` to folder `models`
>  - In the `datasets.py` class a method to apply the Data Normalization step (the `pre_process` method contained in `src/utils`) is missing 
>  - As it was done in `forests.py` we can add the `IsolationForest` class as a sub class of `ExtendedIsolationForest` so that we can easily use also the IF model on our experiments. 

> [!success] Updated `dataset.py` class 
>  - Added instance variable `feature_names`
>  - Added `normalize` and `pre_process` methods
>  - Added `split_dataset` method for the experiments 

> [!done] Added `IsolationForest` class
>  Added the `IsolationForest` class in `EIF_reboot.py`. This class is simply a subclass of `ExtendedIsolationForest` and it modifies two methods: 
>  - In `__init__()` we set `plus` to `False` because the`IF+` model does not exist. 
>  - In `fit()` the `locked_dims` parameter is set to `np.arange(X.shape[1],dtype=int)` because all the dimensions are locked for the computation of the normal hyperplane vectors. 

> [!question] Add DIFFI Implementation to `IsolationForest` ?
>  The newly created `IsolationForest` class inherits from `ExtendedIsolationForest` the methods `local_importances` and `global_importances`. Since ExIFFI is a generalization of DIFFI if I put in input to ExIFFI an `IsolationForest` model does it turn into DIFFI? Theoretically it should.   

- [x] Reboot and review experiments code -> write on a Python script 
	- [ ] Create a result table with all the time execution details to see how the model scales with the dataset size. Compare Isolation Forest with EIF and EIF+ and other AD models (e.g. a state of the art AD AutoEncoder and Deep Isolation Forest). Metrics to use for the comparison: AUC ROC Score, Average Precision, Precision, Recall, F1 Score, ... and the time values,`real_time`,and `user_time`)

> [!done] `test_exiffi` and `test_model` Experiment scripts 
>  I have started to write the backbone of the Python Scripts to use to launch experiments (they are both contained in folder `experiments`):
>  - `test_exiffi` → This script computes the GFI scores of the ExIFFI model and plots the Bar Plot, Score Plot and Importance Map to have a graphical representation of the Feature Importances computed by a specific version of the ExIFFI model. It is in fact possible to modify , through command line arguments, the following parameters: 
> 	 - `n_trees` → Number of trees in the forest 
> 	 - `distribution` → The distribution used to sample the intercept point `p` used in the cutting hyperplanes
> 	 - `eta` → The scale factor used in `distribution`
> 	 - `scaler` → The Scaler used for the Data Normalization step
> 	 - `contamination` → Contamination factor in the dataset
> 	 - `depth_based` → Use the Depth Based version of the importance calculation or not
> 	 - `n_runs` → Number of runs used for the Global Importance computation 
> - `test_model` → This script is used to compare the IF, EIF, EIF+, DIF and AutoEncoder models using the `collect_performance_df` function that computes a so called Performance Report including all the typical metrics used in classification problems. The most important ones for Anomaly Detection are Average Precision and ROC AUC Score. Also in this case there are several parameters to tune in the experiments:
> 	- `contamination`
> 	- `n_trees` → Only for IF, EIF and EIF+
> 	- `n_runs`
> 	- `hidden_neurons` → Only for AutoEncoder
> 	- `distribution` → Only for EIF and EIF+
> 	- `eta` → Only for EIF and EIF+



- [x] Test the implementation of DIF,Autoencoder ,(INNE?) of PyOD. See [here](https://pyod.readthedocs.io/en/latest/pyod.models.html) 

> [!attention] 
>  In notebook `Test_AD_Models.ipynb` I am doing some tests using the new `collect_performance_df` method to compare the EIF, EIF+, DIF and AutoEncoder model. In particular these tests are used to test the PyOD implementation of the DIF and AutoEncoder models. For some reason the EIF and EIF+ models are getting lower Average Precision values with the respect to the ones we reported in the first version of the paper (the Average Precision experiments are collected in the notebook `Average_Precision.ipynb`). Moreover the DIF and AutoEncoder model seems very bad. 

> [!important] Maybe I understand where the error is
>  I found out that in `wine` the `y` variable obtained loading the data has 10 ones at the beginning and than all zeros. On the other hand in pre processing the data with the `preprocess` function I did:
> 

``` python 
y_train=np.zeros(X_train.shape[0])
y_test=np.ones(X_test.shape[0])
y=np.concatenate([y_train,y_test])   
```

> [!important] 
> So I put the 10 ones at the end of the `y` variable → so I changed the order !!!! Fix this thing and see if the performances get back to the correct values.  
> So when I do the tests I have to use `partition_data()` to get the training set `X_train` (only inliers) but then for the test set I do not have to use `X_test=np.r_[X_train,X_test]` but I have to normalize `X` and then use `dataset.y` as the label variable. 

^b1eceb
	
- [ ] Search a good dataset for discussing the results (think about what kind of experiments to do) with ground truth labels where there is some domain knowledge. We want anomalies to be truly isolated points and not just minority classes in a Multi Class Classification problem. → Some possible examples are [[ExIFFI PAPER REVIEW#Benchmark Datasets|here]]. 

- [ ] Ablation studies of EIF+
	- [ ] Different normalizing distributions (i.e. different scalers other than `StandardScaler` → `MinMaxScaler`,)
	- [ ] Different number of Isolation Trees 
	- [ ] Different distribution for selecting the intercept point $p$ from which the partition hyperplane has to pass 
	- [ ] Try to use median instead of mean to select the intercept point (i.e. $N(median(X),\eta \ std(X)$)
	- [ ] Divide in train and test and change the percentage of anomalies in the training set 

- [ ] Ablation studies ExIFFI 
	- [ ] What happens to explainability as the contamination factor increases ? 
	- [ ] Different number of Isolation Trees 
	- [ ] Include or exclude the `depth_based` parameter 
	- [ ] How interpretation changes giving in input only inliers or datasets with also outliers 
	- [ ] Try to use other percentages (other than 10%) to split the dataset in inliers and outliers when we compute the Global Importance 

- [ ] Deep Isolation Forest and compare it to ExIFFI -> say at the end that the problem with Deep Isolation Forest is that is very difficult to apply interpretability to it. 
	- [ ] Focus more on comparing the feature importance obtained with ExIFFI with the one obtained with Random Forest (train the random forest model with the anomaly scores obtained from the Anomaly Detection models).

- [ ] Search for papers on Feature Importance in Decision Trees/Random Forest and see what kind of experiments they did 

- [ ] Interpretable models evaluation metrics from *Evaluation of post-hoc interpretability methods in time-series classification*
	- [ ] Use TIC Index from the paper *Evaluation of post-hoc interpretability methods in time-series classification* 
	- [ ] Use the metrics $AUC\tilde{S}_{top}$ and $F1\tilde{S}$ to evaluate the feature importance

- [ ] Enlarge the Related Work section 
	- [ ] See some possible paper to cite [[ExIFFI PAPER REVIEW#Papers for Related Work|here]]

- [x] User Study → In the review they proposed to conduct a [[ExIFFI PAPER REVIEW#^e08017|User Study]] to evaluate the effectiveness of the Interpretability method. This is something a bit complicated to do so, as Gian suggested, we should cite a paper (that Gian will provide us) that described the possible challenges in including such a study and then we say that we decided to use the Feature Selection proxy task (and possibly the metrics defined in paper *Evaluation of post-hoc interpretability methods in time-series classification*) for the evaluation of the Interpretability part. 

> [!check] 
> Check *Towards A Rigorous Science of Interpretable Machine Learning* 

# Detailed Scheme 

## For 15 February  

- [x] <span style="color:red;">Ask Gian for inclusion of Francesco optimized code in the paper</span>
	- [x] There is also Davide Sartor optimized code to take inspiration from

- [x] Organize better the code to optimize it.
	- [x] Written `make_importance` function in C with `OpenMP` → up to 130 times faster  
	- [x] <span style="color:red;">Add the new features inserted in the plot functions for inserting DIFFI in PyOD</span>
	- [x]  <span style="color:green;">Reboot and review old code</span> 
- [x] <span style="color:red;">Test the implementation of DIF,Autoencoder ,(INNE?) of PyOD</span>

- [ ] <span style="color:yellow;">Search a good dataset for discussing the results (think about what kind of experiments to do) with ground truth labels where there is some domain knowledge. We want anomalies to be truly isolated points and not just minority classes in a Multi Class Classification problem. → Some possible examples are </span>[[ExIFFI PAPER REVIEW#Benchmark Datasets|here]]  

- [x] <span style="color:yellow">Draw a scheme for how to run the experiments (take inspiration also on the studies/experiments of other papers) → also for the Ablation Studies</span> → See [[EXPERIMENT SCRIPTS]]. 

## For 22 February

- [x] <span style="color:red">Reboot and review experiments code -> write on a Python script</span>
	- [x] <span style="color:red">Create a result table with all the time execution details to see how the model scales with the dataset size. Compare Isolation Forest with EIF and EIF+ and other AD models (e.g. a state of the art AD AutoEncoder and Deep Isolation Forest). Metrics to use for the comparison: AUC ROC Score, Average Precision, Precision, Recall, F1 Score, ... and the time values,`real_time`,and `user_time`)</span>
- [x] Add a `contamination` parameter in `ExtendedIsolationForest` → all the PyOD models have one. Moreover it would also help in the `_predict` function instead of having to pass it as an input parameter. 

- [ ] <span style="color:green;">Adapt experiments to the `numba` code </span> 
	- [ ] <span style="color:green;">Add experiments on different versions of `X_train` and `X_test` (e.g start with no anomalies in `X_train` and anomalies in `X_test` and then add some anomalies in `X_train`)</span>
- [x] <span style="color:green;">Implement new version of Feature Selection experiment</span>

> [!done] 
> In the Feature Selection Experiment we sligthly modify the design of the Feature Selection Proxy Task used in the first version of the paper:
> - `direct` → As in the first version of the paper at every iteration the least important feature is removed from the dataset and we plot the behavior of the Average Precision metric as the number of feature decreases
> - `inverse` → The features are removed in the inverse order: from the most to the least important. Also in this case we plot the Average Precision as the number of features decreases.
> 
> At the end two curves are produced, one for `direct` and one for `inverse`, and the higher is the AUC between the two curves the better is the interpretability power of the model. 
> In fact we expect the `direct` curve to decrease slowly then the `inverse` curve, so it should always stay higher then the `inverse` one. 
> 

- [x] <span style="color:red;">Check in DIF Paper if they apply a Data Normalization → maybe Data Normalization is not a good idea before applying all the non linear transformation they apply on the *deep network part* of the DIF model.</span>

> [!done] 
> Both in DIF and in the AutoEncoder model of PyOD the normalization phase is automatically applied in the `fit` method so we do not have to add it. In particular:
> - In DIF the `MinMaxScaler` is used
> - In AutoEncoder there is a boolean parameter `preprocessing` that checks weather to apply normalization (with `StandardScaler`) or not. It is set to `True` by default so normalization is applied. We can however set it to `False` if we want to try to normalize the data with another scaler (e.g. `MinMaxScaler`). 

- [ ] <span style="color:red;">Try to work again on `Test_AD_Models` to get result that have sense on DIF and other models</span>

> [!attention] 
> See [[TO DO ExIFFI Paper Review#^b1eceb|here]] 

- [x] <span style="color:red;">Produce a summary of how the new version of the paper should look like so that we can have a clear idea of how to structure the experiments:</span>
	- [x] <span style="color:red;">create a new note with the paper structure and experiment structure → take inspiration from the old paper structure and from the review.</span>
> [!note] 
> See [[PAPER ORGANIZATION|here]] 

- [ ] Modify the method `collect_performance_df` so that it does not only return a `pd.DataFrame` with just the Performance Report but also with all values of the parameters (so also the value of `n_trees`, `n_runs`, `distribution`, `eta`,...) → take inspiration from the method used in HPC Project (save result in `npz` file after having launched `test_model.py` and then use the `load_stats` method to create a `pd.DataFrame` with the stats of each experiment)

- [x] Create a new branch `plot` to update the `plot.py` script 
	- [x] Essentially move here all the methods I inserted inside the class `Extended_DIFFI_parallel` to produce the plot (i.e. from `compute_local_importances` onward)
	
> [!done]
> Inserted all the new plot functions inside `utils_reboot/plots.py`

- [ ] Open a new branch `datasets` and add some new features on `datasets.py` as described in [[TO DO ExIFFI Paper Review#^4d4de5|here]]

- [ ] Always on a new branch create a wrapper of class `ExtendedIsolationForest` to implement the `IsolationForest` model. 

## TO DO from 27/02 

- [ ] Finish the Importance + Feature Selection Experiments 
	- [ ] Remove `abs` from the $AUC_{FS}$ computation (put a minus in the Feature Selection plots where the blue line is always higher than the blue line)
	- [ ] Add the casual Feature Selection 
- [ ] EIF+ Ablation Study → Do the precision contamination experiments (re organize the folders to do these experiments)
- [ ] Add time computation on the experiments 
- [ ] List of final results to put on a piece of paper → Alessio 