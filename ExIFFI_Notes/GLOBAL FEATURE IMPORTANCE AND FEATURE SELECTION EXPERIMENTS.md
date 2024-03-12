
# Experiment Configurations

> [!note] Hyper parameters
>  - `models = [IF,EIF,EIF+]`
>  - `interpretation = [EXIFFI+,EXIFFI,DIFFI,RandomForest]`
>  - `scenario=[1,2]`

> [!note] Notation
>  - `EXIFFI+` → `EXIFFI` interpretation starting from the `EIF+` model. So it means that we use the `global_importances` method of class `EIF_reboot` using an object initialized as `ExtendedIsolationForest(plus=True)`. This is the same as `EIF+_EXIFFI` in the file names of the plots.
>  - `EXIFFI` → `EXIFFI` interpretation using the `EIF` model. So it means that we use the `global_importances` method of class `EIF_reboot` using an object initialized as `ExtendedIsolationForest(plus=False)`. This is the same as `EIF_EXIFFI` in the file names of the plots. 
>  - `DIFFI` → `DIFFI` interpretation of the `IsolationForest` model. In this case we use the `diffi_ib` method contained in `interpretability_module.py` → code by Mattia Carletti (original implementation of `DIFFI`). Here we have to use the `IsolationForest` model from `sklearn.ensemble`. 
>  - `RandomForest` → Post-hoc interpretation algorithm. We train a `RandomForestRegressor` on the Anomaly Scores produced by one model between `EIF+,EIF,IF` and then we compute the importances using the `feature_importances_` method which implements the Feature Importance of Random Forest. 

- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EXIFFI+` and `scenario=2`|`EXIFFI+` and `scenario=2`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EXIFFI+` and `scenario=1`|EXIFFI+` and `scenario=1`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#EXIFFI` and `scenario=2`|EXIFFI` and `scenario=2`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EXIFFI` and `scenario=1`|`EXIFFI` and `scenario=1`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `DIFFI` and `scenario=2`|`IF` , `DIFFI` and `scenario=2`]] → ==completed== (a part the `nan` problem in `cardio` and `ionosphere`)
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `DIFFI` and `scenario=1`|`IF` , `DIFFI` and `scenario=1`]] → ==completed== (`cardio` has an `inf` feature importance in Feature 5 in the Score Plot)
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF+` , `RandomForest` and `scenario=2`|`EIF+` , `RandomForest` and `scenario=2`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF+` , `RandomForest` and `scenario=1`|`EIF+` , `RandomForest` and `scenario=1`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF` , `RandomForest` and `scenario=2`|`EIF` , `RandomForest` and `scenario=2`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF` , `RandomForest` and `scenario=1`|`EIF` , `RandomForest` and `scenario=1`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `RandomForest` and `scenario=2`|`IF` , `RandomForest` and `scenario=2`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `RandomForest` and `scenario=1`|`IF` , `RandomForest` and `scenario=1`]] → ==completed==

## TODO 

- [x] Finish completely the experiments for `annthyroid`, `glass`, `moodify`
	- [x] Finish contamination plots
	- [x] Complete Feature Selection plots (use all the interpretation models but evaluate the Average Precision with `EIF+` and `EIF`


> [!question] What does this mean?
>  In the Feature Selection experiments there are two components:
>  - Interpretation Model (parameter `--model_interpretation`)→ This is the model that, using the global importances, gives us an order of the importances of the features according to a certain interpretation (e.g. `EIF+,EXIFFI+` gives a certain order, `EIF,EXIFFI` gives a certain order, `IF,DIFFI` gives a certain order, `IF,RandomForest` gives a certain order,...). The global importances (from which we then get the order with `feat_order = np.argsort(matrix.mean(axis=0))`) are contained in paths like `dataset_name/experiments/global_importances/model_name/interpretation_name/scenario_1(or scenario_2)`.I have already computed all these files in the previous experiments and I load them in `test_feature_selection.py` doing `most_recent_file = get_most_recent_file(path_experiment_feats)`. 
>  - Evaluation Model (parameter `--model`) → Once we have the feature order we have to produce the Feature Selection plot computing the Average Precision value with all the features, all the features minus the least/most important (`inverse` or `direct`  strategy) and so on. To have a fair comparison we need to use the same AD model to compute the Average Precision values on these different subset of features → this is what the evaluation model is doing. We will use two possible evaluation models (i.e. `EIF+` and `EIF`). This model is the one I use in these two commands: 
>  ``` python
> 	 direct = feature_selection(I, dataset, feat_order, 10, inverse=False, random=False)
> 	 inverse = feature_selection(I, dataset, feat_order, 10, inverse=True, random=False)
> ```
> For how I designed the Feature Selection experiments (in the `test_GFI_FS.py` script) I have always used the same model as Evaluation and Interpretation model.
> So, looking at the list above I have already done the Feature Selection plots for the following combinations:
> 
> - `scenario=1,EIF+,EXIFFI+`
> - `scenario=2,EIF+,EXIFFI+`
> - `scenario=1,EIF,EXIFFI`
> - `scenario=2,EIF,EXIFFI`
> - `scenario=1,EIF+,RandomForest`
> - `scenario=2,EIF+,RandomForest`
> - `scenario=1,EIF,RandomForest`
> - `scenario=2,EIF,RandomForest`
> 
> The combinations I am missing are: 
> 
> - `scenario=1,EIF+,(EIF,EXIFFI)`
> - `scenario=2,EIF+,(EIF,EXIFFI)`
> - `scenario=1,EIF,(EIF+,EXIFFI+)`
> - `scenario=2,EIF,(EIF+,EXIFFI+)`
> - `scenario=1,EIF+,(IF,DIFFI)`
> - `scenario=2,EIF+,(IF,DIFFI)`
> - `scenario=1,EIF,(IF,DIFFI)`
> - `scenario=2,EIF,(IF,DIFFI)`
> - `scenario=1,EIF+,(IF,RandomForest)`
> - `scenario=2,EIF+,(IF,RandomForest)`
> - `scenario=1,EIF,(IF,RandomForest)`
> - `scenario=2,EIF,(IF,RandomForest)`
> - `scenario=1,EIF+,(EIF,RandomForest)`
> - `scenario=2,EIF+,(EIF,RandomForest)`
> - `scenario=1,EIF,(EIF+,RandomForest)`
> - `scenario=2,EIF,(EIF+,RandomForest)`
>


> [!important] Alternatives for very big datasets → `diabetes, shuttle, moodify`
>  Running the experiments using the complete dataset for these very big datasets takes too much time. We can try two solutions to reduce the execution time:
>  - Downsample the datasets to a smaller number of elements (e.g. `anthyroid` has 7200 samples and does not require and enormous amount of time) → 7000 could be a good sample size
>  - Provide a mini batch of the dataset to each tree (e.g. 2000 samples for each different tree). Obviously these batches should all be different because at the end we are already using subsamples of 256 elements for each tree (extracted from the complete dataset). 
## `EIF+` , `EXIFFI` and `scenario=2` old 

### Synthetic Dataset

- `Xaxis` → ==OK==
	- Global Importance 40 run no pre process → `22-02-2024_22-30` → this makes sense
	- Feature Importance → `22-02-2024_22-32` → makes sense 
	- Global Importance 40 run pre process → `22-02-2024_22-44` → very similar to no pre process (probably because the training data are centered on the origin and so they are already pre processed )
	- Feature Importance → `22-02-2024_22-47` → makes sense 
- `Yaxis`: → ==OK== 
	- Global Importance 40 run pre process → `22-02-2024_22-49` → ok
	- Feature Selection → `22-02-2024_22-50` → ok
- `bisect`: → ==OK== 
	- Global Importance 40 run pre process → `22-02-2024_22-53` → there is a little piece of green (but that's probably because we used 40 runs and so there is the chance of getting the unlucky one)
	- Feature Selection → `22-02-2024_22-54` → ok
- `bisect_3d`: → ==OK== 
	- Global Importance 40 run pre process → `22-02-2024_22-56` → ok
	- Feature Selection → `22-02-2024_22-56` → ok → sligthly worse than `bisect` but that's because there are more anomalous features and so the interpretation task is more difficult 
- `bisect_6d`: → for some reason it doesn't see the `csv` file in the path 
	- Global Importance 40 run pre process → `22-02-2024_22-56`
	- Feature Selection → `22-02-2024_22-50` 

> [!important] 
>  The performances are probably a bit worse than the ones obtained in the first paper version because in the Scenario 2 we are doing:
>  `dataset.split_dataset(train_size=0.8,contamination=0)` so with `contamination=0` we are taking only the inliers (so as usual for Scenario 2 we train only on the inliers) but doing `train_size=0.8` we are taking a subsample of the inliers (80% of them) so we are essentially training on less data. 
>  In case we want to take all the inliers we have to use `train_size=1-p` where `p` is the contamination factor of the dataset. 
### Real World Datasets

- `wine` :
	- Global Importance 10 run → `22-02-2024_21-38`
	- Feature Selection → `22-02-2024_21-53` → no sense → maybe it's the dataset that is bad 
	- Global Importance 40 run → `22-02-2024_22-02`
	- Feature Importance → `22-02-2024_22-07`
	- Global Importance 40 run con tutti gli inlier → `24-02-2024_10-02` → does not change a lot 
	- Feature Importance → `24-02-2024_10-45` → Also added the value of $AUC_{FS}$ on the plot  
- `glass`
	- Global Importance 40 run → `23-02-2024_08-59` → not so good 
	- Feature Selection → `23-02-2024_09-04` → very bad, but also in the previous version of the Feature Selection plot the one of `glass` was very bad
- `cardio`
	- Global Importance 40 run → `22-02-2024_23-23`
	- Feature Selection → `22-02-2024_23-28` → balena shape 
- `pima`
- `breastw`
- `ionosphere`
- `annthyroid`
	- Global Importance 40 runs → `23-02-2024_09-07` → not perfectly 100% Feature 1 but that's probably because of some unlucky run. 
	- Feature Selection → `23-02-2024_09-09` → Strangely the `direct` selection increases from 2 to 1 features. The `inverse` one instead continuously increases but starts from low Precision values so the $AUC_{FS}$ will not be so huge. In any case the increasing behavior of the Average Precision curve is similar to the first version of the paper. 
- `pendigits`
	- Global Importance 40 runs → `23-02-2024_09-14` → Similarly to the paper the results are not so good 
	- Feature Selection → `23-02-2024_09-19` → The `inverse` direction has an up and down path similar to the one seen in the first version of the paper. 
- `diabetes`
	- Global Importance 40 runs → `23-02-2024_10-01` → Ok, but redo it setting `f=4` in the `bar_plot` function 
	- Feature Selection → `23-02-2024_10-11` → Similar to old paper 
- `shuttle`
	- Global Importance 40 runs → `23-02-2024_10-22` → disaster
	- Feature Selection → `23-02-2024_10-30` → even more disaster
- `moodify`
	- Global Importance 40 runs → `23-02-2024_11-20` not good (`loudness` instead of `energy` at the first place)
	- Feature Selection → `23-02-2024_12-32` → Strangely the `inverse` path increases from 3 features onwards. 

> [!note] 
> Added new folder `plots_new` for each dataset (then the folder plots can be deleted and we rename the folder `plots_new` as `plots`). 
> - Restarting all the experiments changing `train_size` in `split_dataset` to `train_size=1-dataset.perc_outliers` so that we have all the inliers on the training set
> - Running again the experiments because I added the visualization of the $AUC_{FS}$ value on the Feature Selection plot 

> [!todo] 
> Re do `annthyroid` and `wine` since I have added the `feature_names` attribute 

##  `EXIFFI+` and `scenario=2`

#### Synthetic Datasets

- `Xaxis` → ==ok==
- `Yaxis` → ==ok==
- `bisect` → ==ok==
- `bisect_3d` → ==ok==
- `bisect_6d` → ==ok==

#### Real World Dataset

- `wine` → ==ok==
- `glass` → ==ok==
- `cardio` → ==ok==
- `pima` → ==ok==
- `breastw` → ==ok==
- `ionosphere` → ==ok==
- `annthyroid`  → ==ok==
- `pendigits` → ==ok==
- `diabetes` → ==ok==
- `shuttle` → ==ok==
- `moodify` → ==ok==

## `EXIFFI+` and `scenario=1`

### Synthetic Dataset 

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok

### Real World Dataset 

- `wine`  → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok 
- `breast` → ok 
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok 
- `diabetes`→ ok
- `shuttle` → ok 
- `moodify` → ok

## `EIF+`, `EXIFFI+` and `scenario=2`

## `EIF+`, `EXIFFI+` and `scenario=1`

##  `EXIFFI` and `scenario=2`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok

#### Real World Dataset

- `wine` → ok
- `glass`  → ok 
- `cardio` → ok
- `pima` → ok
- `breastw` → ok
- `ionosphere` → ok 
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes` → ok 
- `shuttle` → ok
- `moodify` → ok

## `EXIFFI` and `scenario=1`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok 
- `breast` → ok
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes`→ ok 
- `shuttle` → ok 
- `moodify` → ok


## `EIF`, `EXIFFI+` and `scenario=2`

## `EIF`, `EXIFFI+` and `scenario=1`

## `IF` , `DIFFI` and `scenario=2`

### Real World Datasets
### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ==ok==
- `glass` → ok
- `cardio` → error → `nan` values in importance computation 
- `pima` → ok 
- `breast` → ok, Feature 8 with importance `inf`
- `ionosphere` → error → `nan` values in importance computation 
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes`→ ok
- `shuttle` → ok
- `moodify` → ok

## `IF` , `DIFFI` and `scenario=1`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok
- `glass` → ok
- `cardio` → ok, Feature 5 with importance `inf`
- `pima` → ok
- `breast` → ok
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes`→ ok
- `shuttle` → ok
- `moodify` → ok

## `EIF+` , `RandomForest` and `scenario=2`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok 
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok 
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breast` → ok 
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok 
- `diabetes`→ Job 201510 → ok, relocate box in feature selection plot
- `shuttle` → Job 201510 → ok
- `moodify` → Feature Selection → Job 201578 → 

## `EIF+` , `RandomForest` and `scenario=1`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breast` → ok
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok 
- `diabetes`→ Job 201491 → ok, relocate the box in feature selection plot
- `shuttle` → Job 201491 → ok 
- `moodify` → Job 201529 → only Bar and Score Plot done 
	- Feature selection → Job 201577 → 
## `EIF` , `RandomForest` and `scenario=2`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breast` → ok
- `ionosphere` → ok 
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes`→ Job 201495 → Done 2 times (take one of the two) → relocate box in feature selection plot 
- `shuttle` → Job 201495 → Done 2 times (take one of the two) → relocate box in feature selection plot 
- `moodify` → Job 201531 → ok 

## `EIF` , `RandomForest` and `scenario=1`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breastw` → ok
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes`→ Job 201496 → ok
- `shuttle` → Job 201496 → ok 
- `moodify` → Job 201530 → ok

## `IF` , `RandomForest` and `scenario=2`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breast` → ok
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes`→ Job 201526 → ok, relocate the box in the feature selection plot
- `shuttle` → Job 201526 → ok 
- `moodify` → Job 201532 → ok 

## `IF` , `RandomForest` and `scenario=1`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok
### Real World Datasets

- `wine`  → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok 
- `breastw` → ok
- `ionosphere` → ok
- `annthyroid` → ok
- `pendigits` → ok
- `diabetes` → Job 201520 → ok, relocate box in feature selection plot 
- `shuttle` → Job 201520 → ok
- `moodify` → Job 201533 → ok, relocate box in feature selection plot 

## `pickle` file to update

- `cardio`:
	- `IF/DIFFI/scenario_2` → problem with `nan`
- `ionosphere` :
	- `IF/DIFFI/scenario_2` → problem with `nan`
- `moodify`: 
	- `EIF/RandomForest/scenario_1` →CAPRI
	- `EIF/RandomForest/scenario_2` →CAPRI
	- `EIF+/RandomForest/scenario_1` →CAPRI
	- `EIF+/RandomForest/scenario_2` →CAPRI

## Problem with `pima,ionosphere` and `breastw` datasets

`pima`, `ionosphere` and `breastw` have an high contamination factor: 34.89%, 35.71% and 52.56% respectively. In particular `train_size`=0.8 + these contamination percentages is higher than 1.

In the case of `pima` we call `dataset.split_dataset(train_size=0.8,contamination=0)` in Scenario 2. So essentially we want to create a training set of only inliers taking a subsample containing 80% of the inliers of the original dataset.

> [!example] `pima`
> `pima` has 768 samples, so 80% of the inliers is 614.4. We round this number to 614.

> [!important] 
> The problem is that if we divide `dataset.y` into the inliers and outliers indexes we have 268 outliers and 500 inliers. So we can't take 614 inliers from the original dataset. 

> [!example] `ionosphere`
> In case of `ionosphere` the inliers are 225 but with `train_size=0.8` we try to take 280 inliers from the original dataset. 

> [!example] `breastw`
> In case of `breastw` the inliers are 213 and we try to take 359 inliers with `train_size=0.8`. 

So essentially the problem is that `len(indexes_inliers) < dim_train`

In the case I have `contamination=c` as a parameter of `split_dataset` the problem appears when `len(indexes_inliers) < (1-c)*dim_train`

So if we have `contamination=0` then it should hold that `len(indexes_inliers) >= dim_train`. 

On the other hand if we have `contamination=c`, `split_dataset` works if `len(indexes_inliers) >= (1-c)*dim_train`. So if

$$
	c \leq \frac{dimtrain - len(inliers)}{dimtrain}
$$
> [!done] 
> The solution is than in general we want to have `train_size <= 1-p`  so in `split_dataset` if the argument `train_size` passed is higher than `1-p` we automatically set it to `1-p`. In this case the resulting `dataset.X_train` will contain all the inliers of the dataset. 

# Final Experiments Global Feature Importance 

## `Xaxis` → ==ok== 

- `scenario=1`
	- `EIF+`
		- `random`
		- Ad-hoc 
			- `EXIFFI+` → ==ok==
			- `EXIFFI` → ==ok==
			- `DIFFI` → ==ok==
		- Post-hoc (surrogate model)
			- `EIF+_RandomForest` → ==ok==
			- `EIF_RandomForest` → ==ok==
			- `IF_RandomForest` → ==ok==
	- `EIF`
		- `random` → ==ok==
			- Ad-hoc 
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok== 
			- Post-hoc (surrogate model)
				- `EIF+_RandomForest` → ==ok==
				- `EIF_RandomForest` → ==ok== 
				- `IF_RandomForest` → ==ok== 

- `scenario=2`
	- `EIF+`
		- `random` → ==ok==
		- Ad-hoc 
			- `EXIFFI+` → ==ok== 
			- `EXIFFI` → ==ok==
			- `DIFFI` → ==ok==
		- Post-hoc (surrogate model)
			- `EIF+_RandomForest` → ==ok==
			- `EIF_RandomForest` → ==ok==
			- `IF_RandomForest` → ==ok==
	- `EIF`
		- `random` → ==ok==
			- Ad-hoc 
				- `EXIFFI+` → ==ok== 
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+_RandomForest` → ==ok==
				- `EIF_RandomForest` → ==ok==
				- `IF_RandomForest` → ==ok== 
## `bisect` → ==ok==

- `scenario=1`
	- `EIF+`
		- `random`
		- Ad-hoc 
			- `EXIFFI+` → ==ok==
			- `EXIFFI` → ==ok==
			- `DIFFI` → ==ok==
		- Post-hoc (surrogate model)
			- `EIF+_RandomForest` → ==ok==
			- `EIF_RandomForest` → ==ok==
			- `IF_RandomForest` → ==ok==
	- `EIF`
		- `random` → ==ok==
			- Ad-hoc 
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+_RandomForest` → ==ok==
				- `EIF_RandomForest` → ==ok==
				- `IF_RandomForest` → ==ok==

- `scenario=2`
	- `EIF+`
		- `random` → ==ok==
		- Ad-hoc 
			- `EXIFFI+` → ==ok== 
			- `EXIFFI` → ==ok==
			- `DIFFI` → ==ok==
		- Post-hoc (surrogate model)
			- `EIF+_RandomForest` → ==ok==
			- `EIF_RandomForest` → ==ok==
			- `IF_RandomForest` → ==ok==
	- `EIF`
		- `random` → ==ok==
			- Ad-hoc 
				- `EXIFFI+` → ==ok== 
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+_RandomForest` → ==ok==
				- `EIF_RandomForest` → ==ok==
				- `IF_RandomForest` → ==ok==

## `glass` → ==ok== 

- `scenario=1`
		- `EIF+`
			- `random` → use `--compute_random` in the first test of `EIF+, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==

	- `scenario=2`
		- `EIF+`
			- `random` → use `--compute_random` in the first test of `EIF+, scenario=2` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `EXIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
				- `DIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
				- `EIF, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
				- `IF, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=2` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
				- `EXIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
				- `DIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
				- `EIF, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot
				- `IF, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot

## `glass_DIFFI`

`glass_DIFFI` is the version of the `glass` dataset used in the `DIFFI` paper. It uses the data from the `headlamps` glass class as the outliers → so now we have 29 outliers. On the other hand in `glass` (the version we took from the UCI Machine Learning Repository) the `tableware` glass class was the one used for the outliers → 9 outliers. In the Bar and Score Plots we were able to reproduce the results obtained in the `DIFFI` paper with Feature 7 (Barium) being clearly the most important one. Now probably also the Feature Selection plots should look better than the one of `glass`, that had [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#^ccccf7|this problem]]. 

- `scenario=1`
	- `EIF+`
		-  `random` → use `--compute_random` in the first test of `EIF+, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok== 
			- Post-hoc (surrogate model)
				- `EIF+,RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+,RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==
		- `IF`
			- `random` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → 
				- `EIF, RandomForest` → 
				- `IF, RandomForest` → 

	- `scenario=2`
		- `EIF+`
			- `random` → use `--compute_random` in the first test of `EIF+, scenario=2` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok== 
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+,RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=2` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+,RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==
		- `IF`
			- `random` → 
			- Ad-hoc
				- `EXIFFI+` → 
				- `EXIFFI` → 
				- `DIFFI` → 
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → 
				- `EIF, RandomForest` → 
				- `IF, RandomForest` → 

## `annthyroid` → ==ok==

- `scenario=1`
	- `EIF+`
		- `random` → use `--compute_random` in the first test of `EIF+, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok== 
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok==
				- `EIF, RandomForest` → ==ok==
				- `IF, RandomForest` → ==ok==

	- `scenario=2`
		- `EIF+`
			- `random` → use `--compute_random` in the first test of `EIF+, scenario=2`  → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `EXIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `DIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `EIF, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `IF, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=2` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `EXIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `DIFFI` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== (redone with `train_size=1-dataset.perc_outliers`) → does not change a lot 
				- `EIF, RandomForest` → ==ok== 
				- `IF, RandomForest` → ==ok==
## `moodify` → ==ok==

- `scenario=1`
	- `EIF+`
		 - `random` → use `--compute_random` in the first test of `EIF+, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok== 
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== 
				- `EIF, RandomForest` → ==ok== 
				- `IF, RandomForest` → ==ok==
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=1` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok==
				- `EXIFFI` → ==ok== 
				- `DIFFI` → ==ok==
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== 
				- `EIF, RandomForest` → ==ok== 
				- `IF, RandomForest` → ==ok== 
		
	- `scenario=2`
		- `EIF+`
			- `random` → use `--compute_random` in the first test of `EIF+, scenario=2` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok== (right)
				- `EXIFFI` → ==ok== (right)
				- `DIFFI` → ==ok== (right)
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== (right)
				- `EIF, RandomForest` → ==ok== (right)
				- `IF, RandomForest` → ==ok== (right)
		- `EIF`
			- `random` → use `--compute_random` in the first test of `EIF, scenario=2` → ==ok==
			- Ad-hoc
				- `EXIFFI+` → ==ok== (right)
				- `EXIFFI` → ==ok==
				- `DIFFI` → ==ok== (right)
			- Post-hoc (surrogate model)
				- `EIF+, RandomForest` → ==ok== (right) 
				- `EIF, RandomForest` → ==ok== (right)
				- `IF, RandomForest` → ==ok== (right)
# Comments on the results 

Let's look carefully at all the plots produced for each dataset and get some conclusions (that will then go in the paper)

## Synthetic Datasets 

In the synthetic datasets the anomalies are very evident and they should be easy to detect and interpret for `Xaxis` and `Yaxis` (since there is only one "anomalous" feature) while the task becomes more difficult with `bisect_3d` and `bisect_6d` where the number of "anomalous" feature increases. 

### `Xaxis`

- `EIF+`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Here Feature 0 is clearly considered the most important with high margin on all the others 
		- Feature Selection Plot: Here the area between the two feature selection strategies is pretty wide → $AUC_{FS} = 4.615$ 
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Feature 0 is still considered the most important in the Score Plot but with a smaller margin and moreover it is not 100% of the times the most important in the Bar Plot
		- Feature Selection Plot: $AUC_{FS} = 2.926$ → the area is more or less half than the one in scenario 2
	- `RandomForest, scenario=2`
		- Bar/Score Plots: Here the dominance of Feature 0 is even more evident in the Score Plot (the Bar Plot is more or less the same as in `EXIFFI, scenario=2`). That's because we use the `EIF+` scores as labels for a `RandomForest` regressor that than has a probably more robust way to compute the Feature Importances. Obviously the problem of `RandomForest` is that it needs labels.
		- Feature Selection Plot: Very similar to `EXIFFI, scenario=2` → $AUC_{FS} = 4.539$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Also in this case in the Score Plot Feature 0 is the most important but with a lower margin on the others while in the Bar Plot it is still at the first position in all the 40 runs
		- Feature Selection Plot: Like for `EXIFFI, scenario=1` the area between the red and blue line reduces to more or less half of the previous value → $AUC_{FS} = 2.92$
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Similar to `EXIFFI, scenario=2` with a clear dominance of Feature 0 on the others
		- Feature Selection Plot: Similar to `EIF+, EXIFFI, scenario=2` with even a sligthly better $AUC_{FS}$ value → $AUC_{FS} = 4.733$. That is probably because we have still a single anomalous feature. `EIF+` should be better in cases of multiple "anomalous" features.
    - `EXIFFI, scenario=1` 
	    - Bar/Score Plots: As in `EXIFFI; scenario=1` the dominance of Feature 0 sligthly decreases but it is still the most important feature. 
	    - Feature Selection Plot: $AUC_{FS} = 3.438$ → lower than in `scenario=2` but higher than the one of `EIF+, EXIFFI, scenario=1`
	- `RandomForest, scenario=2`
		- Bar/Score Plots: As in `EIF+, RandomForest, scenario=2` the dominance of Feature 0 is very clear.
		- Feature Selection Plots: Similar to `EXIFFI, scenario=2` → $AUC_{FS} = 4.706$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Dominance of Feature 0 is still there but sligthly less evident
		- Feature Selection Plot: $AUC_{FS}$ decreases → $AUC_{FS} = 3.48$
- `IF`
	- `DIFFI, scenario=2` 
		- Bar/Score Plots: Very bad. The importance scores are assigned more or less randomly. Probably if we launch another time the experiment the feature order may change. The Score Plot is not very informative because all the importance values are very similar and there is Feature 1 that is considered the most important. This probably happens because in scenario 2 we train only on the inliers and the `IF` model probably needs to see some outliers in order to work well. 
		- Feature Selection Plots:  $AUC_{FS} = 1.438$ → the red line has a peak with 2 features (that's probably because 0 and 1 have are sligthly more important than the others considered together?) and then the precision drops going from 2 to 1 feature. 
      - `DIFFI, scenario=1`  
		  - Bar/Score Plots: In scenario 1 the results are much better and resemble the typical results seen in scenario 2 for the previous configurations with Feature 0 raising as the most important feature. This result confirms the fact that `IF` requires to have some contamination in the training set to work well. So also for the rest of the datasets we should expect to have better results in `scenario=1`  for the `IF` model. 
		  - Feature Selection Plot:  $AUC_{FS} = 2.568$ Now the plots makes sense → the blue line is more or less always at the same low Precision values and the red one increases Precision as we remove the least important features. 
	 - `RandomForest, scenario=2` 
		 - Bar/Score Plots: The results are still quite random but Feature 0 is the most important in almost 60% of the cases in the Bar Plot and it is first in the Score Plot (not by a lot but still first). 
		 - Feature Selection Plot: The Feature Selection plot is much better → $AUC_{FS} = 2.458$
	 - `RandomForest, scenario=1` 
		 - Bar/Score Plots: In this case passing to scenario 1 does not help `IF` to achieve better results. The importance of Feature 0 and 1 is very similar. 
		 - Feature Selection Plot: $AUC_{FS} = 1.523$ → similarly to `IF,DIFFI scenario=2` → there is a peak at 2 features. 

> [!error] Problem in `IF, RandomForest` Experiments 
> In the function `compute_global_importances` when `interpretation='RandomForest'` we fit a `RandomForestRegressor` on the Anomaly Scores of the AD Model we are using. In the test script we have always used the `sklearn` implementation of the `IF` model because we need to compute the feature importances with `DIFFI`. The problem generates when we do `rf.fit(dataset.X, I.predict(dataset.X))`. In fact the `predict` method of `sklearn.ensemble.IsolationForest` returns the labels → -1 for outliers and +1 for inliers. If we want to obtain the Anomaly Scores we have to use the `I.decision_function(dataset.X)`. There is another problem here: the Anomaly Scores computed by `sklearn.ensemble.IsolationForest`  are different than the ones computed by us in `EIF_reboot` or the ones computed by the `IForest` implementation of `PyOD`. 
>  

> [!success] Solution
>  There are two possible solutions:
>  - In case the interpretation is `RandomForest` and the model is `IF` we have to use the implementation of `IF` contained in `EIF_reboot`. So the `sklearn` version of `IF` is used only when `interpretation=DIFFI` 
>  - Use the `sklearn` implementation of `IF` but the fit of the `RandomForestRegressor` should be done using the Anomaly Scores of `IF` → `rf.fit(dataset.X, I.decision_function(dataset.X))`
> Probably the second solution is better because `sklearn.ensemble.IsolationForest` is faster than our implementation. 
>  **Re Run all the `IF, RandomForest` experiment after this change**
>  

### `Yaxis`

Here the results should be very similar to the ones of [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`Xaxis`|`Xaxis`]] but with Feature 1 as the most important one. 

- `EIF+`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: As expected the most important feature (with margin) is Feature 1
		- Feature Selection Plot: Very high area as in `Xaxis` → $AUC_{FS} = 4.584$
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Feature 1 is still the most important but with less margin on the other features. 
		- Feature Selection Plots: $AUC_{FS}$ decreases to $AUC_{FS} = 2.917$
    - `RandomForest, scenario=2` 
	    - Bar/Score Plots: As it happened in `Xaxis` Feature 1 is the most important one with an even higher margin
	    - Feature Selection Plots: Similar to `EXIFF, scenario=2` → $AUC_{FS} = 4.558$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Feature 1 is still the most important but with less margin on the other features.
		- Feature Selection Plot: $AUC_{FS}$ decreases to $AUC_{FS} = 2.859$
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: As expected the most important feature (with margin) is Feature 1 → similar to `EIF+, EXIFFI, scenario=2`
		- Feature Selection Plot: Similar to `Xaxis` → $AUC_{FS}$ is higher than the one of `EIF+, EXIFFI, scenario=2` → $AUC_{FS} = 4.721$
	 - `EXIFFI, scenario=1`
		 - Bar/Score Plots: Feature 1 is still the most important but with less margin on the other features.
		 - Feature Selection Plot: $AUC_{FS}$ decreases to $AUC_{FS} = 3.54$
	- `RandomForest, scenario=2`
		- Bar/Score Plots: Dominance of Feature 1 is very clear as in `EIF+, RandomForest, scenario=2`
		- Feature Selection Plot: Similar to `Xaxis` → $AUC_{FS}$ is higher than the one of `EIF+, EXIFFI, scenario=2` → $AUC_{FS} = 4.734$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Feature 1 is still the most important but with less margin on the other features.
		- Feature Selection Plot: $AUC_{FS}$ decreases to $AUC_{FS} = 3.359$
- `IF` 
	- `DIFFI, scenario=2` 
		- Bar/Score Plots: Very random as in `Xaxis`
		- Feature Selection Plots: Similar to `Xaxis` → $AUC_{FS} = 2.555$
	- `DIFFI, scenario=1` 
		- Bar/Score Plots: As in `Xaxis` with scenario 1 the situation is better and Feature 1 is clearly the most important one
		- Feature Selection Plots: Similar to `Xaxis` → $AUC_{FS} = 2.633$
	- `RandomForest, scenario=2` 
		- Bar/Score Plots: Similarly to `Xaxis` the results are more or less random but with a slight advantage of Feature 1. 
		- Feature Selection Plots: Similar to `Xaxis` → $AUC_{FS} = 2.565$
	- `RandomForest, scenario=1` 
		- Bar/Score Plots: Pretty random. As in `Xaxis, IF, RandomForest, scenario=1` passing from `scenario=2` to `scenario=1` does not improve the results.  
		- Feature Selection Plot: $AUC_{FS} = 2.614$

### `bisect`

> [!note] 
>  In all the configurations, except for `EIF+, EXIFFI, scenario=1` the $AUC_{FS}$ metric decreases from `scenario=2` to `scenario=1`. The differences are not huge but still this is a different behavior than the one observed in `Xaxis` and `Yaxis`

- `EIF+`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Good, there is not an exactly 50/50 division in the Bar Plot between Feature 0 and Feature 1 but in the Score Plot it is clear that these two features are dominant with the respect to the others. 
		- Feature Selection Plot: Good but the $AUC_{FS}$ value is a bit lower than the one obtained in `Xaxis` and `Yaxis` because the blue line does not immediately drop to almost 0 precision. In fact since there are 2 important features once we remove the most important one in the `direct` Feature Selection strategy there is still the other one that makes the precision not so bad (e.g. in this case when we pass from 6 to 5 features the precision drops from $\approx 0.9$ to $\approx 0.6$. On the other hand in `Xaxis, EIF+, EXIFFI, scenario=2` it drops from $\approx 0.9$ to $\approx 0$) → $AUC_{FS} = 4.076$
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: As usual as we pass from scenario 2 to scenario 1 the Bar and Score Plots are sligthly worse. In fact the the importance of Feature 0 and 1 has less margin with the respect to the other features. 
		- Feature Selection Plot: The $AUC_{FS}$ metric decreases to $AUC_{FS} = 3.901$
	- `RandomForest, scenario=2`
		- Bar/Score Plots: The importance of Feature 0 and 1 is more clear here (as it also happened with the other datasets) with a slight advantage of Feature 0 → that's probably due to some randomness in the runs. 
		- Feature Selection Plot: The $AUC_{FS}$ metric decreases to $AUC_{FS} = 3.943$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: As usual still Feature 0 and 1 are the most important but with less margin
		- Feature Selection Plot: Strangely $AUC_{FS} = 3.966$ → sligthly higher than `scenario=2`
- `EIF` 
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: We can still see Feature 0 and 1 emerging as more important than the others but with a lower margin then with `EIF+, EXIFFI` or `EIF+, RandomForest` 
		- Feature Selection Plot: $AUC_{FS} = 4.145$ → sligthly higher than `EIF+, EXIFFI`
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Here we have a perfect 50/50 division in the Bar Plot and very similar importance values in the Score Plot
		- Feature Selection Plot: $AUC_{FS} = 4.181$ → sligthly higher than `scenario=2` 
	- `RandomForest, scenario=2`
		- Bar/Score Plots: Here Feature 0 is always the most important one and Feature 1 always the second one and there is a significant difference in their importance scores. The importance values of the other features are much smaller. 
		- Feature Selection Plot: Similar to the other cases → $AUC_{FS} = 4.042$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Now we are closer to a 50/50 between Feature 0 and Feature 1 and in fact their importance values are closer in the Score Plot. 
		- Feature Selection Plot: $AUC_{FS} = 4.125$ → sligthly higher than `scenario=2`
- `IF`
	- `DIFFI, scenario=2` 
		- Bar/Score Plots: As usual `scenario=2` is pretty random. 
		- Feature Selection Plot: $AUC_{FS} = 4.028$ Pretty high → we have this strange V shape in the red line between 3 and 1 feature. 
	- `DIFFI, scenario=1` 
		- Bar/Score Plots: Better as usual, clear division of importance between Feature 0 and Feature 1. 
		- Feature Selection Plot: $AUC_ {FS} = 4.436$ → Better than `scenario=2`. Now the red line continues to increase as we remove features. 
	- `RandomForest, scenario=2`
		- Bar/Score Plots: As usual `RandomForest` tends to detect a single feature as the most important one so here Feature 0 and 1 have much higher importances values than the other features but the higher importance is assigned to Feature 0. 
		- Feature Selection Plot: $AUC_{FS} = 4.45$ → Very high → also higher than `EIF+, EXIFFI, scenario=2`
	- `RandomForest, scenario=1` 
		- Bar/Score Plots: Similar to `scenario=2` with a smaller difference in importances between Feature 0 and 1. 
		- Feature Selection Plot: Similar to `scenario=2` → $AUC_{FS} = 4.469$ 

### `bisect_3d`

Now the anomalies are distributed along 3 features so the task of identifying correctly the anomalies is harder and we should probably see lower $AUC_{FS}$ values in the Feature Selection plots. 

- `EIF+`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: The importance is almost equally shared between the first 3 features (e.g. Feature 0, 1, 2) 
		- Feature Selection Plot: Following the reasonement done in `bisect, EIF+, EXIFFI, scenario=2` now we have 3 important features and so three steps to drop from $\approx 0.9$ to $\approx 0$ precision. This reduces the $AUC_{FS}$ value to $AUC_{FS} = 3.376$. 
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Similar to `scenario=2` but with smaller margin between the important and non important features 
		- Feature Selection Plot: As it happened also in `bisect` here the $AUC_{FS}$ metric increases to $AUC_{FS} = 3.804$
	- `RandomForest, scenario=2` 
		- Bar/Score Plots: Here Feature 2 is always the most important one and it has an importance value significantly higher than the ones of Feature 0 and 1 in the Score Plot. 
		- Feature Selection Plot: The $AUC_{FS}$ value increases to $AUC_{FS} = 3.405$ with the respect to `EIF+, EXIFFI, scenario=2`. In particular we can notice the fact that there is a clear dominance of Feature 2 because in the red line (representing the `inverse` Feature Selection strategy) there is a V shape in the last 3 steps (so passing from 3 to 2 features the precision decreases and increases back again passing from 2 to 1). 
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Now the importance score are more evenly shared between the three features.
		- Feature Selection Plot: The $AUC_{FS}$ increases to $AUC_{FS}=3.684$. This happens because the blue line (i.e. `inverse` Feature Selection strategy) decreases more smoothly and slowly to a low precision value than what happens in `scenario=2`
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Similar to `EIF+, RandomForest, scenario=1`. 
		- Feature Selection Plots: $AUC_{FS} = 3.411$
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: More or less similar to `scenario=1`
		- Feature Selection Plot: $AUC_{FS}$ increases to $AUC_{FS} = 3.812$
	- `RandomForest, scenario=2`
		- Bar/Score Plots: Very similar to `EIF+, RandomForest, scenario=2` with Feature 2 as clearly the most important feature. 
		- Feature Selection Plot: $AUC_{FS} = 3.445$ 
	- `RandomForest, scenario=1`
		- Bar/Score Plots: As in `EIF+, RandomForest, scenario=1` the importance is now more evenly shared among Feature 0, 1 and 2. 
		- Feature Selection Plot: $AUC_{FS}$ increases to $AUC_{FS} = 3.803$
- `IF`
	- `DIFFI, scenario=2` 
		- Bar/Score Plots: Pretty random importance values → more or less same importance on all the features. 
		- Feature Importance Plot: $AUC_{FS} = 1.83$ → The red line has a similar behavior to the blue one between 6 and 4 features → there is a V shape because the precision decreases and then start to increase. 
	- `DIFFI, scenario=1` 
		- Bar/Score Plots: Now the importance is more concentrated on the three most important features 0,1 and 2. 
		- Feature Importance Plot: Better Feature Selection plot → $AUC_{FS} = 3.946$
	- `RandomForest, scenario=2`: 
		- Bar/Score Plots: As in `EIF+, RandomForest, scenario=2` Feature 2 is the most important
		- Feature Selection Plot: $AUC_{FS} = 3.964$ → very high value. All the precision values in the `inverse` red line are really close to 1. 
	- `RandomForest, scenario=1` 
		- Bar/Score Plots: Feature 2 is still the most important one but with a smaller margin on Feature 0 and 1. 
		- Feature Importance Plot: $AUC_{FS} = 3.956$ → The `direct` line takes sligthly more time to go down and so the $AUC_{FS}$ value is lower than in `scenario=2`. 

### `bisect_6d`

In this dataset all the features are important (since the anomalies are distributed among all the 6 features)

- `EIF+`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: All multi color bars and essentially equal scores for all the features in the Score Plot (as expected)
		- Feature Selection Plot: Here the Feature Selection Plot is very bad in terms of the $AUC_{FS}$ metric $AUC_{FS} = -0.022$. However different from the other cases here the precision does not drop to 0. Here in fact all the features are important. The precision values continuously decrease up to the step where we pass from 2 features to 1 where the precision increases → probably in this dataset having a single feature provides better performances than having multiple features. 
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Similar to `scenario=2`
		- Feature Selection Plot: Similar to `scenario=2` → $AUC_{FS} = -0.026$
	- `RandomForest, scenario=2`
		- Bar/Score Plots: As usual `RandomForest` always finds a feature that is significantly more important than the others, Feature 2 in this case. Here the second and third most important features are 3 and 0 respectively. In fact in this case all the features are important so we are not forced to have the first 3 (0,1,2) as the most important ones.  
		- Feature Selection Plot: The shape is similar to the one already seen with `EXIFFI`. However the $AUC_{FS}$ value is higher (and positive) because having a dominant important feature makes the red `inverse` line always higher than the blue one (except for the case where we have 2 features) → $AUC_{FS} = 0.121$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: More even importance scores distribution but still Feature 2 has a significant margin on the others. 
		- Feature Selection Plot: Very similar to `scenario=2` → $AUC_{FS} = 0.053$
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Similar to `EIF+, EXIFFI, scenario=2`
		- Feature Selection Plot: Similar to the others already seen → $AUC_{FS} = 0.124$
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Similar to `scenario=1`
		- Feature Selection Plot: Similar to `scenario=1` → $AUC_{FS} = 0.001$
	- `RandomForest, scenario=2`
		- Bar/Score Plots: `RandomForest` finds Feature 2 as the most important as usual → 3 and 0 are the second a third most important
		- Feature Selection Plot: $AUC_{FS} = 0.24$
	- `RandomForest, scenario=1` 
		- Bar/Score Plots: Similar to `scenario=2`, also in this case Feature 2 is considered the most important 
		- Feature Selection Plot: $AUC_{FS} = 0.18$
- `IF`
	- `DIFFI, scenario=2` 
		- Bar/Score Plots: Pretty random scores as seen also in the other configurations. 
		- Feature Selection Plots: $AUC_{FS} = -0.03$
	- `DIFFI, scenario=1` 
		- Bar/Score Plots: Similar to `scenario=2`
		- Feature Selection Plot: $AUC_{FS} = 0.83$
	- `RandomForest, scenario=2`
		- Bar/Score Plots: `RandomForest` finds Feature 2 as the most important as usual → 3 and 0 are the second a third most important
		- Feature Selection Plot: $AUC_{FS} = 0.232$
	 - `RandomForest, scenario=1`
		- Bar/Score Plots: `RandomForest` finds Feature 2 as the most important as usual → less margin on the other two differently from `scenario=2`
		- Feature Selection Plot: $AUC_{FS} = 0.241$

> [!note] 
> It seems that in `RandomForest` there is always a single feature raising out as the most important, also in the cases of `bisect` and `bisect_3d` where the importance is shared between multiple features. This is a point in favor of `EXIFFI` that is instead able to correctly assign  a shared importance score to multiple important features. 

## Real World Datasets

### `wine`

> [!note] What to expect
>  In the first edition of the paper `wine` had not very high value for the Precision scores → in Scenario 1 it had [0.22,0.22,0.18] for IF,EIF,EIF+ respectively and [0.4,0.58,0.78] in Scenario 2. 
>  For what concerns the Feature Importances we have a clear dominance of Feature 12 (`Proline`) in Scenario 2 while Scenario 1 was more random (very strange thing is that in Scenario 1 Feature 12 is the least important one while the two most important ones are Feature 4 and 8 (8 is the last one in Scenario 2)). In the Feature Selection Plot the Precision increases from 0.2 to 0.8 as we remove less important features. 
>

- `EIF+`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Here we have a clear advantage of Feature 12 over the others even though it is not the most important one 100% of the times in the Bar Plot. 
		- Feature Selection Plot: The red line increases from a bit higher than 0.2 to a bit lower than 0.8 as expected. The blue line instead drops immediately to values around 0.1 → $AUC_{FS} = 4.685$
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Very random (even more random than in the previous version). The most important features in the Score Plot are `Magnesium` and `Proanthocyanins` with `Proline` going down to $9^{th}$ place. 
		- Feature Selection Plot: Here the situation is very bad → $AUC_{FS} = -0.697$ since the blue line is higher then the red one most of the time. Here it happens the same thing described in `RandomForest, scenario=1` → as soon as we remove `Proline` (the $5^{th}$) least important feature the precision drops. 
	- `RandomForest, scenario=2` 
		- Bar/Score Plots: As in the other cases `RandomForest` is very good in identifying a single dominant important feature in fact it is very clear that `Proline` is the most important one. 
		- Feature Selection Plot: The shape is more or less the same as `EXIFFI, scenario=2` but there is a strange increase in the blue line going from 5 to 3 features, so $AUC_{FS} = 2.848$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Now we have a significant dominance of feature `Proanthocyanins` and `Magnesium` is at second place. 
		- Feature Selection Plot: As in the case of `EXIFFI, scenario=1` the score is negative → $AUC_{FS} = -2.27$. It seems that in `scenario=1` it is better to remove everytime the most important feature. In any case the peak in Precision is obtained with the two least important features that, looking at the Score Plot, are `Proline` and `Flavanoids` → the fact that there is `Proline` is very strange → maybe there is an error? Or we can interpret this result in the following way → with `EXIFFI+` and `RandomForest, scenario=2` we have  found out that the real important feature is `Proline` however with `scenario=1` it is not possible to see the importance of `Proline` (because putting the anomalies in the training set the model thinks that they are normal points?) 


> [!important] Riflessione risultati `wine`
>  Ho scoperto una cosa un pò particolare guardando i plot di `wine`. Allora:
>  -  Nella prima versione del paper noi abbiamo visto che nello Scenario 2 la feature più importante è chiaramente la 12 (che d'ora in poi chiamiamo `Proline`) mentre nello scenario 1 ci sono valori un po random (e tra l'altro `Proline` è addirittura all'ultimo posto). 
>  - Con `EXIFFI+, scenario=2` si conferma il fatto che `Proline` è la più importante e nel plot Feature Selection tutto funziona come atteso con la linea rossa sempre sopra la blu e che aumenta al diminuire del numero di feature. 
>  - Con `EXIFFI+, scenario=1` invece gli score ritornano ad essere molto randomici e `Proline` finisce al nono posto (5 feature meno importante). Nel Feature Selection plot abbiamo un valore di `AUC_FS`  negativo perchè per la maggior parte del tempo la linea blu sta sopra alla rossa a parte da quando passiamo da 5 a 4 feature → da li in poi passa sopra la linea rossa. E qual'è la quinta feature meno importante (che quando viene tolta fa scendere a picco la precisione)? Proprio `Proline`. Una cosa molto simile succede con `RandomForest`. 
>  - `RandomForest, scenario=2` → Qua si vede bene che la Feature 12 è la più importante (in particolare anche negli altri casi si vede che `RandomForest` funziona molto bene quando bisogna individuare una singola feature come più importante) e il Feature Importance plot funziona come atteso (simile a `EXIFFI+, scenario=2`).
>  - Passando a `RandomForest, scenario=1` di nuovo abbiamo un valore di `AUC_FS` negativo perchè la linea blu sta sopra alla rossa. Il picco di precisione nella linea blu si presenta quando ci sono solo 2 feature. E qual'è la second feature meno importante nello Score plot (che quando viene tolta fa cadere a picco la precisione)? Sempre il nostro amico `Proline`. 
>  
>Quindi concludendo secondo me questi risultati ci fanno capire come `Proline` sia realmente la feature più importante. Lo Scenario 2 riesce ad individuarla chiaramente come quella più importante mentre nello scenario 1 (probabilmente perchè si inseriscono le anomalie nel training set) non si riesce a metterla come più importante e quindi funziona meglio la Feature Selection `direct` perchè la togliamo verso la fine (mentre su quella `inverse` si toglie subito). 


- `EIF`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: `Proline` clearly is the most important feature but with less margin on the others than with `EXIFFI, scenario=2`
		- Feature Selection Plot: $AUC_{FS} = 3.591$ → similar to `EXIFFI+, scenario=2`
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: Similarly to `EXIFFI+, scenario=1` now the most important features are `Magnesium` and `Proanthocyanins` and `Proline` goes down to $6^{th}$ place ($8^{th}$ least important feature)
		- Feature Selection Plot: As expected the peak in the blue line goes down passing from 8 to 7 features (so after we remove `Proline`). In this case `Proline` is remove pretty soon in the `direct` approach so $AUC_{FS}$ is positive → $AUC_{FS} = 0.709$
	- `RandomForest, scenario=2`
		- Bar/Score Plots: `Proline` most important feature, not with a so high margin as it happens in `EIF+, RandomForest, scenario=2`.
		- Feature Selection Plot: $AUC_{FS} = 3.318$. As previously we have a strange increase in precision in the blue line from 6 features to 2 features. 
	- `RandomForest, scenario=1`
		-  Bar/Score Plots: Random importance scores. Most important features are now `Proanthocyanins`, `Phenols` and `Magnesium`
		- Feature Selection Plot: $AUC_{FS} = -2.192$. As usual we have a drop in precision passing from 3 to 2 features (so when we remove `Proline` that is the $3^{rd}$ least important feature)
- `IF` 
	- `DIFFI, scenario=2` → ==review after the new experiments==
	- `DIFFI, scenario=1` → ==review after the new experiments==
	- `RandomForest, scenario=2`
		- Bar/Score Plots: Here we have `Phenols` that is the most important feature, followed by `Proline`
		- Feature Selection Plot: $AUC_{FS} = 3.077$ → it works as expected but we see a drop in precision (in the red line) as we pass from 2 to 1 features (so when we remove `Proline`). This result let us conclude about the superiority of `EIF/EIF+` over `IF` because these two models were able to detect `Proline` as the most important feature in Scenario 2 while `IF` not. 
	- `RandomForest, scenario=1` → ==review after the new experiments==

### `cardio`

> [!note] What to expect
>  The precision values are are a bit higher. In Scenario 1 we have (for IF, EIF, EIF+ respectively) [0.58,0.56,0.53] and for Scenario 2 instead we have [0.71,0.74,0.78].
>  For what concerns the importances, Feature 6 is clearly the most important one (detected in Scenario 2) while in Scenario 1 the two most important ones are Feature 2 and 6. 
>  The trend in the Feature Selection plot is more or less constant and it starts to increase from 8 features onward and it decreases a little bit from 3 features onward. 

- `EIF+`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Feature 6 is considered the most important but not with a so high margin on the others (it is first in around 50/60% of the cases in the Bar Plot)
		- Feature Selection Plot: $AUC_{FS} = 6.991$. The red line has a behavior more or less similar to the one of the previous version of the paper. The difference is that we have a peak in precision passing from 5 to 4 features (i.e. so when we remove Feature 2) and then precision drops again passing from 4 to 3. 
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: All pretty random importance values in the Bar Plot and in the Score plot we have Feature 2 and 6 as the most important ones. 
		- Feature Selection Plot: $AUC_{FS} = 4.615$ The red line has a decreasing trend with peaks on 8 and 4 features. 
	- `RandomForest, scenario=2`
		- Bar/Score Plots: As usual `RandomForest` is good in detecting a single important feature, in fact here Feature 6 has a dominant importance score over all the others. 
		- Feature Selection Plot: $AUC_{FS} = 6.984$ 
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Very strange. It is pretty clear that Feature 16 and 17 are the most important ones. Feature 6 ends up in $13^{th}$ place. 
		- Feature Selection Plot: $AUC_{FS} = 0.95$ → Considering that there are 21 features in `cardio` (and only the first 15 of them are represented in the Score Plot) this means that Feature 6 is the $9^{th}$ least important feature. In fact the drop in precision happens when we pass from 9 to 8 features. 
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Clearly Feature 6 is the most important. 
		- Feature Selection Plot: $AUC_{FS} = 7.172$. Similar shape to `EXIFFI+` and there is a V shape passing from 3 to 1 features → precision increases a lot passing from 2 to 1 features → so considering Feature 6 alone we have a good precision. 
	- `EXIFFI, scenario=1`
		- Bar/Score Plots: More random values and the most important features are 2 and 6. 
		- Feature Selection Plot: $AUC_{FS} = 5.464$ Red line has a decreasing shape. As expected the precision decreases a lot when we pass from 2 to 1 feature (so when we remove Feature 6)
	- `RandomForest, scenario=2`
		- Bar/Score Plots: As usual Feature 6 is the most important with an high margin on the others. 
		- Feature Selection Plot: $AUC_{FS} = 8.027$ Better result then `EXIFFI+, scenario=2`. The shape of the plot is very similar to `EXIFFI+` but the precision values are higher (that's why the $AUC_{FS}$ metric is higher). Probably this happens because `RandomForest` detects Feature 6 as the most important with more confidence? 
	- `RandomForest, scenario=1`
		- Bar/Score Plots: As in `EIF+, RandomForest, scenario=1` we have Feature 16 and 17 as the most important ones, Feature 6 is still at $13^{th}$ place  
		- Feature Selection Plot: $AUC_{FS} = -0.071$ Big drop from 5 to 4 features in the blue line.
- `IF`
	- `DIFFI, scenario=2` → ==review after the new experiments==
	- `DIFFI, scenario=1` → ==review after the new experiments==
	- `RandomForest, scenario=2`
		- Bar/Score Plots: Feature 17 is the most important one with high margin on the others
		- Feature Selection Plot: $AUC_{FS} = 6.743$ 
	- `RandomForest, scenario=1` → ==review after the new experiments==

### `glass`

For `glass` I computed all the 24 Feature Selection plots as described in [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#Final Experiments Global Feature Importance|here]] so here I can report a complete description and comments on the results obtained. 

> [!note] What to expect
>  In the first version of the paper both in `scenario 1` and `scenario 2` the most important features were Feature 5 and Feature 4 (followed by Feature 6 in `scenario 1` and by Feature 7 in `scenario 2`). For what concerns the Feature Selection proxy task the Average Precision values are pretty low (more or less between 0.1 and and 0.2/0.25). In `EXIFFI+` there is an increase in the Average Precision starting from 4 features onwards. 

Let's start from an analysis of only the Bar and Score Plots, then we will look separately at the Feature Selection Plots so that we can compare the ad-hoc and post-hoc interpretations. 

- `EIF+`
	- `EXIFFI+, scenario=2`
		- Bar Plot: The importance is divided between Features 4,5,6,7 in the first three ranks.
		- Score Plot: In fact the first 4 positions in the Score Plot are 5,4,6,7 respectively. 
	- `EXIFFI+, scenario=1`
		- Bar Plot: Similar to `scenario 2` but Feature 5 is at the first place more than 50% of the times in the Bar Plot (so it is a significant advantage on the others). 
		- Score Plot: We can see that Feature 5 has a bit higher importance score than the others
	- `RandomForest, scenario=2`
		- Bar Plot: Here there is clearly something strange going on. Apparently there is Feature 2 that is the most important one in all the runs. As usual `RandomForest` is good in detecting a dominant important feature but here it is choosing a different one then the one expected. 
		- Score Plot:
	- `RandomForest, scenario=1`
		- Bar Plot:
		- Score Plot:

> [!attention] Problem `glass`
>  Allora ho fatto tutti i nuovi plot Feature Selection per `glass`, `annthyroid` e ora sto facendo quellli di `moodify`. 
>  Stavo un attimo guardando i plot di `glass` e mi sono accorto di una cosa. 
>  
>  - Con le interpretazioni `EXIFFI, EXIFFI+` e `DIFFI` nei Bar e Score Plot si vede che le Feature più importanti sono la 5 e la 4 (seguite a breve distanza da Feature 6 e Feature 7) con valori di importanza quasi uguali. 
>  - Nelle interpretazioni con `RandomForest` invece viene la Feature 2 come nettamente la Feature più importante (con ampio margine sulle altre). 
>  
>  Quindi ho pensato che `RandomForest` stesse sbagliando completamente e invece se guardiamo i plot della Feature Selection sono nettamente meglio quelli di `RandomForest`. 
>  
>  Infatti i plot con le interpretazioni ad-hoc (`EXIFFI+,EXIFFI,DIFFI`) hanno la linea blu sopra la rossa in molti punti (in alcuni casi ci sono anche dei zig zag cioè prima la blu sopra e poi la rossa sopra) e hanno spesso valori di `AUC_FS` negativi. Mentre i plot con `RandomForest` come modelli di interpretazione hanno sempre la linea rossa sopra e quindi sempre valori positivi di `AUC_FS`. 
>  Da questo quindi si dovrebbe concludere che per `glass` la interpretazione con `RandomForest` è migliore rispetto alle altre. 

^ccccf7
### `glass_DIFFI`

Since the [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`glass`|results of the `glass` dataset]] are a bit disappointing we used the version of the `glass` dataset used in the DIFFI paper which uses 29 anomalies (glasses from the `headlamps glass` class instead of the `tableware glasses` used as anomalies in the UCI Machine Learning Repository version). Let's comment here the results of this new version of `glass`.

#### Importance Plots

- `EIF+`
	- `EXIFFI+, scenario=2`
		- Bar Plot: Here we have a confirmation of the results obtained in the DIFFI paper. As expected (also from domain knowledge) the most important feature is Barium (`Ba` in the plot, Feature 7). 
		- Score Plot: `Ba` is the most important feature, followed by `K` (Potassium). 
	- `EXIFFI+, scenario=1`
		- Bar Plot: The importances are randomly splitted across all the features. 
		- Score Plot: Similar importance value across all the features: `K` and `Ca` on top. 
	- `RandomForest, scenario=2`
		- Bar Plot: Strange: Feature 2 (`Mg` Magnesium) is the most important one in all the runs. As usual `RandomForest` is good in detecting a dominant important feature but here it is choosing a different one then the one expected. 
		- Score Plot: `Mg` is the most important feature and, as it usually happens in `RandomForest`, it has a very significant margin on the others. `Ba` is at third place, after `Na`. 
	- `RandomForest, scenario=1`
		- Bar Plot: Also in scenario 1 `RandomForest` gives the highest importance to `Mg`. 
		- Score Plot: `Mg` still on top but with a smaller margin. 
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar Plot: Similar to `EXIFFI+, scenario=2` with `Ba` as the most important feature in all the runs. 
		- Score Plot: `Ba` on top followed by `K`. 
	- `EXIFFI, scenario=1`
		- Bar Plot: Still pretty random but there is a slight advantage of `K` (it is the most important feature in 50/55% of the runs). 
		- Score Plot: The two most important features are `K` and `Ca` with a slight advantage on the others. 
	- `RandomForest, scenario=2`
		- Bar Plot: Similar to `EXIFFI+, scenario=2` → `Mg` clearly on top
		- Score Plot: Similar to `EXIFFI+, scenario=2` → `Mg` clearly on top, `Ba` in second to last position ($8^{th}$ place)
	- `RandomForest, scenario=1` 
		- Bar Plot: Similar to `EXIFFI+, scenario=1` → `Mg` clearly on top
		- Score Plot: Similar to `EXIFFI+, scenario=1` → `Mg` on top but with a sligthly lower margin on the others
- `IF` → Here we should get very similar results to the ones obtained in the `DIFFI` paper 
	- `DIFFI, scenario=2`
		- Bar Plot: `Ba` on top as in `EXIFFI+` and `EXIFFI`
		- Score Plot: `Ba` on top with a huge margin on the others (38.6 vs 3.352)
	- `DIFFI, scenario=1`
		- Bar Plot: Differently from `EXIFFI+` and `EXIFFI` here `Ba` maintains its importance also in `scenario 1`
		- Score Plot: `Ba` still on top, followed by `K` with a small margin. 
	- `RandomForest, scenario=2`
		- Bar Plot: Here we have still `Mg` as clearly the most important feature but there is `Ba` at the second place. 
		- Score Plot: Here we have still `Mg` as clearly the most important feature but there is `Ba` at the second place. 
	- `RandomForest, scenario=1`
		- Bar Plot: Here the situation completely changes: the most important features are `Si` and `Ca`.
		- Score Plot: The most important features are `Si` and `Ca` while `Mg` and `Ba` end up at $7^{th}$ and $9^{th}$ place. 
#### Feature Selection Plots 

Here we divide the Feature Selection plots into the ones where the Average Precision values were evaluated with `EIF+` and the ones evaluated with `EIF`. Fixing one evaluation model (`EIF+` or `EIF`) we want to see which interpretation algorithm is the best. In particular we want also to compare the ad-hoc interpretability method (`EXIFFI+`, `EXIFFI`, `DIFFI`) with the post-hoc interpretation obtained with `RandomForest`. Let's analyse separately `scenario 1` and `scenario 2`.

> [!note] 
>  In the plots where the blue line is over the red one (so when the metric is negative) it means that the interpretation model has wrongly ranked the most important features in the last places. 

> [!question] 
> May the fact that with 2 features the precision stays the same mean that these two features are containing very similar information? (highly correlated) So having both of them or only one of the them does not change a lot. Or maybe this happpen because  they have similar importance values? 

- `EIF+`
	- `scenario 2`
		- `EIF+`
			- `EXIFFI+` → $AUC_{FS} = 1.637$ → The plot works as expected with the red line increasing. However the blue line stays at more or less a constant level between 8 and 3 features then decreases passing to 2 and increases again going to 1.  
			- `EIF+_RandomForest` → $AUC_{FS} = 0.757$. A significant drop in precision can be observed passing from 3 to 2 features in the red line and from 7 to 6 features in the blue line. This is expected because the $3^{rd}$ most important feature (so the third one to be discarded in the `direct` approach and the third to last one to be discarded in the `inverse` approach) is `Ba`. This result confirms the fact that `Ba` is the most important feature and it was correctly identified as such by the `EXIFFI+` interpretation model. 
		- `EIF`
			- `EXIFFI` → Similar to `EXIFFI+` but now the `direct` approach line does not have that strange V shape at the end. For this reason $AUC_{FS} = 1.682$, sligthly higher than the `EXIFFI+` one 
			- `EIF_RandomForest` → Here we have the result correct but reverse. In the sense that, since in the Score plot of `EIF_RandomForest_2` we have `Ba` at the second to last place we should expect something similar to what we have seen in `EIF+_RandomForest` with the precision drops passing from 8 to 7 and from 2 to 1 features. This happens but, for some strange reason, the blue and red lines are switched. So apparently here the feature that causes the precision drop is the one that is `K` (the one that is in last position in the Score Plot). In any case `K` is the second most important feature in `EXIFFI+` and `EXIFFI` so it still makes sense. 
		- `IF`
			- `DIFFI` → $AUC_{FS} = 1.587$ Similar to `EXIFFI+` and `EXIFFI` but here the `direct` line has a significant decrease from 4 features onward. The shape of the plot looks better but the $AUC_{FS}$ is lower because the Average Precision is lower. In fact in `EXIFFI+` and `EXIFFI` we have the last two points at 0.6 while here we have just the last point at 0.6. So in `EXIFFI+` and `EXIFFI` the first two features together have more or less the same performance as using only the most important one (that is the usual `Ba`). This happens because the first two features in `EXIFFI+` and `EXIFFI` are `Ba` and `K` while the first two in `DIFFI` are `Ba` and `Na`. But this happens because we are evaluating with `EIF+` (for which the most important features are `Ba` and `Na`) → what happens if I evaluate the Average Precision values with `IF` ? 
			- `IF_RandomForest` → $AUC_{FS}=1.405$ In the Score Plot of `IF_RandomForest` `Ba` is the second most important feature and in fact we have a precision drop going from 8 to 7 and from 2 to 1 features in the Feature Selection Plot. This confirms the superiority of `DIFFI` with the respect to the surrogate `RandomForest` model. 

Evaluating the Average Precision with the `IF` model the same trend is confirmed: the $AUC_{FS}$ values is high in `EXIFFI`, then `EXIFFI+` and `DIFFI` is still at the last place. For what concerns the last two points of the `inverse` line we still have similar values in `EXIFFI` and `EXIFFI+` and in `DIFFI` there is a slight increase passing from 2 to 1 feature, but the increase is not as significant as in the case of `EIF+` as an evaluation model. 

To complete the analysis we should also look at what happens with `EIF` as the evaluation model. The $AUC_{FS}$ order is still the same, the strange fact is that, both in `EXIFFI` and `EXIFFI+` passing from 2 to 1 feature the precision sligthly decreases (in particular in the case of `EIF_EXIFFI+`). If we look at the Score Plots for `EXIFFI` and `EXIFFI+` the most important features are in both cases `Ba` and `K` → (in `EXIFFI+`: `Ba` → 2.578, `K` → 2.367; in `EXIFFI`: `Ba` → 3.325, `K` → 2.925). 

- `scenario 1`
	- `EIF+`
		- `EXIFFI+`→ $AUC_{FS} = 0.541$ Here as expected the value is lower than the one obtained in `scenario 2` because the importances were pretty random. In particular in the `inverse` line we have a drop in precision passing from 4 to 3 features so when `Ba` is removed (in fact in the Score Plot of `EXIFFI+, scenario 1` feature `Ba` is at $4^{th}$ place). Then there is an increase in precision passing from 2 features to 1. This makes sense because with 1 features we are using just feature `K` (that is the second most important in `EXIFFI+ scenario 2` , and so it is surely most important than `Ca`). This result is a confirmation of the superiority of `scenario 2` over `scenario 1`.
		- `EIF+_RandomForest` → $AUC_{FS} = -1.436$. In the Score Plot of `EIF+_RandomForest` we have `Ba` at the last place and `K` at $5^{th}$ place. In the Feature Selection plot we have in the `direct` line a decrease in precision when `K` is removed (passing from 5 to 4 features). In the `direct` line instead there is a drop in precision when we remove `Na` and remain just with feature `Mg` (passing from 2 to 1 features). This proves the fact that `Mg` is surely not the most important feature for the `glass_DIFFI` dataset. 
	- `EIF`
		- `EXIFFI` → $AUC_{FS} = 0.551$. Similar to `EXIFFI+, scenario 1`. Big drop in precision passing from 3 to 2 features, so when we remove `Ba` (that is in third place in the Score Plot of `EXIFFI scenario 1` ). Then there is an increase passing from 2 features to 1 features because we are removing `Ca` and using only `K`. 
		-  `EIF_RandomForest` → As said [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#^136e06|here]] $AUC_{FS} = -1.505$ in fact `Ba` is at last place and `K` at third to last place in the Score Plot of `EIF_RandomForest scenario 1`. 
	- `IF`
		- `DIFFI` → As we saw in the Importance plot here `Ba` is still the most important feature so the Feature Selection plot is much better than the others → $AUC_{FS} = 1.535$. In the last two points the precision is around 0.6 → so having 2 features or 1 is pretty much the same. 
		- `IF_RandomForest` → Here as we can see from the Importance plots the situation completely changes and also the most important features change a lot (`K` ends up in third place and `Ba` in the last place) so obviously the Feature Selection plot becomes very bad → $AUC_{FS} = -1.333$. There is a little drop in precision passing from 7 to 6 features in the blue line, so when we remove `K` (this makes sense because `K` is an important feature). In the red line instead there is a significant increase when we pass from 5 to 4, so when we remove `Na`. 

- `EIF`
	- `scenario 2`
		- `EIF+` 
			- `EXIFFI+` → $AUC_{FS} = 1.759$. The plot shows the expected shape a part for the passage from 2 to 1 feature where there is this little drop in precision that is a bit strange. 
			- `EIF+_RandomForest` → $AUC_{FS} = 0.757$. In the Score Plot `Ba` is placed at third place and `K` at $8^{th}$. In fact passing from 8 to 7 features in the `direct` approach and from 3 to 2 features in the `inverse` approach the Average Precision has a significant decrease. 
		- `EIF`
			- `EXIFFI` → Similar to `EXIFFI+`, here the decrease passing from 2 to 1 features is less evident, in fact $AUC_{FS} = 1.934$ → better than `EXIFFI`. 
			- `EIF_RandomForest` → $AUC_{FS} = -1.682$. In the Score Plot `Ba` and `K` are the last two features. So the fact that the two most important features are positioned in the last two places makes the $AUC_{FS}$ negative. As expected in the `direct` approach there is a decrease when we pass from 2 to 1 (i.e. when `Ba` is removed) and also when I remove `Ba` in the `inverse` approach passing from 8 to 7 features. 
		- `IF`
			- `DIFFI` → $AUC_{FS} = 1.771$ Similar to `EXIFFI` and `EXIFFI+`. The Average Precision values in the blue line start to decrease from 4 features onward (so from the moment in which `K` ($6^{th}$ place in the Score Plot) is removed). 
			- `IF_RandomForest` → $AUC_{FS} = 1.405$. Here `Ba` is at the second place in the Score Plot so we have a drop in precision in the `direct` and `inverse` approaches passing from 8 to 7 and from 2 to 1 features respectively (so when `Ba` is removed from the feature space). 


	- `scenario 1`
		 -  `EIF+`
			- `EXIFFI+` → $AUC_{FS} = 0.437$ pretty zig zagging behavior. Drop in precision passing from 4 to 3 features (i.e. after removing `Ba`) and increase in precision passing from 2 to 1 (i.e. passing to just `K` in the feature space) in the `inverse` approach. In the `direct` approach instead there is a little drop when we remove `Ba` (passing from 6 to 5 features). 
			- `EIF+_RandomForest` →  $AUC_{FS} = -1.436$ In the Score Plot `Ba` is in the last place and `K` at $5^{th}$, that's why the $AUC_{FS}$ metric is negative. There is a little drop in precision passing from 5 to 4 features in `direct` approach (i.e. when `K` is removed). In the `inverse` approach passing from 4 to 3 features the precision increase (so when we remove `Fe`) and than starts to decrease again until we get to the last point (only feature `Mg`) where it gets its lowest value. 
		- `EIF`
			- `EXIFFI` → $AUC_{FS} = 0.491$ There is a V shape in the last 3 steps in the `inverse` approach. In fact when we pass from 3 to 2 features we are removing `Ba` and when we pass from 2 to 1 we are removing `Ca` and leaving `K` alone. 
			- `EIF_RandomForest` → $AUC_{FS} = -1.699$ `Ba` is at the last place in the Score Plot and `K` is at the third last place so $AUC_{FS}$ is negative. Also here the precision sligthly increases in the `inverse` approach passing from 4 to 3 features, so when `Fe` is removed. 
		- `IF` 
			- `DIFFI` → $AUC_{FS} = 1.584$ This is the best Feature Selection Plot in `scenario 1` because, as we saw in the Importance Plots, `DIFFI` is the only interpretation model that is able to recognize `Ba` as the most important feature also in `scenario 1`. In particolar in the `inverse` approach there is a significant increase in Average Precision as we pass from 3 to 2 features, so when `Ca` is removed to leave only `Ba` and `K` in the feature space. 
			- `IF_RandomForest` → `Ba` at last place and `K` at third place in the Score Plot so $AUC_{FS} = -1.777$ (negative). Also here we have a little increase in precision when `Fe` is removed in the `inverse` approach (passing from 5 to 4 features). 

### `annthhyroid`

#### Importance Plots

- `EIF+`
	- `EXIFFI+, scenario=2`
		- Bar Plot: Feature 1 most important feature almost 90% of the times. 
		- Score Plot: Feature 1 on top, followed by Feature 3 and 2 (not by a huge margin). 
	- `EXIFFI+, scenario=1`
		- Bar Plot: Feature 1 now is the most important one only 50% of the times. 
		- Score Plot: Feature 1 still on top but with very close values to all the other features except feature 1 that is at the last place with a significant margin. 
	- `RandomForest, scenario=2`
		- Bar Plot: `RandomForest` is good in detecting a single important feature so Feature 1 is first all the times and Feature 2 is second 90% of the times. 
		- Score Plot: Feature 1 on top with a clear margin on Feature 3. 
	- `RandomForest, scenario=1`
		- Bar Plot: Here Feature 3 is the most important more than 90% of the times while Feature 1 is the least important more the 90% of the times. 
		- Score Plot: Feature 3 most important feature with a clear margin on Feature 2 and Feature 1 is clearly at the last place. 
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar Plot: Feature 1 is at the first place in all the runs. 
		- Score Plot: Feature 1 clearly on top, followed by Feature 2. 
	- `EXIFFI, scenario=1`
		- Bar Plot: Feature 1 at first place around 70% of the times. Feature 0 instead is at the last place in all the runs.
		- Score Plot: Feature 1 stil at the first place but with a minimal margin on Feature 5 (that has the same importance value (2.721) as Feature 4). 
	- `RandomForest, scenario=2`
		- Bar Plot: Similar to `EIF+_RandomForest, scenario 2`. Feature 1 is clearly the most important, followed by Feature 3. 
		- Score Plot: Similar to `EIF+_RandomForest, scenario 2`. First Feature 1, second Feature 3. 
	- `RandomForest, scenario=1`
		- Bar Plot: As in `EIF+_RandomForest, scenario 1` Feature 3 now becomes the most important. 
		- Score Plot: Similar to `EIF+_RandomForest scenario 1` with the difference that now Feature 1 is at $5^{th}$ place. 
- `IF`
	-  `DIFFI, scenario=2`
		- Bar Plot: Feature 1 first place in all the runs and Feature 0 at last place in all the runs. 
		- Score Plot: Feature 1 at first place with a small margin on Feature 2. 
	- `DIFFI, scenario=1`
		- Bar Plot: Similar to `scenario 2`.
		- Score Plot: Similar to `scenario 2` but there is an higher margin between Feature 1 and the rest. 
	- `RandomForest, scenario=2`
		- Bar Plot: Feature 1 always first and Feature 3 second almost in all the runs (there was probably 1 or 2 runs where Feature 2 was second).
		- Score Plot: Feature 1 and 3 at the first two ranks. 
	- `RandomForest, scenario=1`
		- Bar Plot: As it already happened in `scenario 1` for `RandomForest` now Feature 3 becomes the most important but Feature 1 is at the second place 60% of the runs. 
		- Score Plot: Feature 3 at first place, Feature 1 at second place. 

#### Feature Selection Plots 

- `EIF+`
	- `scenario_2`
		- `EIF+`
			- `EXIFFI+` → $AUC_{FS} = 1.619$ Standard shape expect for the fact that there is an increase in precision passing from 2 to 1 feature in the `direct` approach (so removing Feature 4 to remain with just Feature 0). 
			- `EIF+_RandomForest` → $AUC_{FS} = 1.516$ Here in the Score Plot `EIF+_RandomForest` is similar to the one of `EXIFFI+` so also the Feature Selection plots are similar. The $AUC_{FS}$ here is a bit smaller because in the `direct` approach there is an increase starting from 3 features onward (so after removing Feature 5 and 4). 
		- `EIF`
			- `EXIFFI` → $AUC_{FS} = 1.493$ Similar to the previous two. The metric value is smaller because also in this case from 3 to 1 features the Average Precision increases in the `direct` approach. 
			- `EIF_RandomForest` → $AUC_{FS} = 1.513$ Very similar to `EXIFFI` (also the two Score Plots are similar), the higher value obtained may be just for some random effects. 
		- `IF`
			- `DIFFI` → Similar to the one of `EXIFFI`, $AUC_{FS} = 1.524$. Also in this case probably some random effects caused the $AUC_{FS}$ value to be sligthly higher than the `EXIFFI` one.
			- `IF_RandomForest` → Very similar to `DIFFI` → $AUC_{FS} = 1.509$ 


	- `scenario 1`
		-  `EIF+`
			- `EXIFFI+` → The shape is very similar to the one of `scenario 1` but the precision values in the `inverse` approach start from smaller values (from 0.2 instead than 0.4). The consequence is that the are between the red and blue line is smaller and thus $AUC_{FS} = 0.938$. The reason under this result is that, looking at the Score Plots, the feature orders in the two scenarios are similar but the importance values are closer in `scenario 1`  so it may happen that Feature 3 has an higher importance than Feature 5 just for some random effects and thus the precision may decrease a little bit in the first steps. When we reach the last step however both plots reach around 0.8 Average Precision. In all the plots seen up to now there is a significant jump in Precision passing from 2 to 1 feature (much higher than what we saw in [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`glass_DIFFI`|`glass_DIFFI`]] for example) and this means that Feature 1 is really dominant in this dataset. 
			- `EIF+_RandomForest` → Here there is a big change because Feature 1 is considered the least important feature so, as expected, $AUC_{FS} = -1.185$ (is negative). 
		- `EIF`
			- `EXIFFI` → Very similar to `EXIFFI, scenario 1` → $AUC_{FS} = 0.93$ 
			- `EIF_RandomForest` → In the Score Plot Feature 1 is placed at the $5^{th}$ place so the plot presents a negative $AUC_{FS} = -0.705$ and in the `direct` approach plot a significant drop in Average Precision can be observed passing from 2 to 1 features (i.e. when Feature 1 is removed from the feature space).  
		- `IF`
			- `DIFFI` → Very similar to `EXIFFI` and `EXIFFI+` → $AUC_{FS} = 0.924$
			- `IF_RandomForest` → Differently from the other cases with `RandomForest` in `scenario 1` here $AUC_{FS} = 0.426$ is positive because Feature 1 is placed at second place so we have a drop in precision passing from 5 to 4 features in the `direct` approach and from 2 to 1 feature in the `inverse` approach. 

- `EIF`
	- `scenario_2`
		- `EIF+`
			- `EXIFFI+` → $AUC_{FS} = 1.818$ → Very similar to `EIF+_EXIFFI+ scenario 2` but  the $AUC_{FS}$ value is higher because the Average Precision values are sligthly higher. Also in this case there is this strange increase from 2 to 1 features in the `direct` approach. 
			- `EIF+_RandomForest` → $AUC_{FS} = 1.729$ → As in `EIF+_EIF+_RandomForest scenario 2` it is very similar to `EXIFFI+` but $AUC_{FS}$ is sligthly smaller because in the `direct` approach the precision starts to increase from 3 features to 1 features. 
		- `EIF`
			- `EXIFFI` → Similar to `EIF+_EXIFFI, scenario 2` → $AUC_{FS} = 1.72$ 
			- `EIF_RandomForest` → Similar to `EXIFFI`.
		- `IF`
			- `DIFFI` → Also here very similar to the previous Feature Selection plots. → $AUC_{FS} = 1.6991$
			- `IF_RandomForest` → $AUC_{FS} = 1.731$


	- `scenario 1`
		-  `EIF+`
			- `EXIFFI+` → As in `EIF+` the plot is very similar in the shape to the corresponding one in `scenario 2` but $AUC_{FS} = 1.012$ is lower because the Average Precision values are lower. 
			- `EIF+_RandomForest` → $AUC_{FS} = -1.211$. As in `EIF+`  $AUC_{FS}$ is negative because Feature 1 is wrongly considered the least important feature. 
		- `EIF`
			- `EXIFFI` → Same story → same shape as `EXIFFI scenario 2` but lower Average Precision values and thus $AUC_{FS} = 0.973$ is lower. 
			- `EIF_RandomForest` → Here like in `EIF+_EIF_RandomForest`  Feature 1 is the $5^{th}$ feature in order of importance and thus there is a negative $AUC_{FS} = -0.707$ with a drop in Precision when we pass from 2 features to 1 (i.e. when Feature 1 is removed). 
		- `IF` 
			- `DIFFI` → Like `DIFFI, scenario 2` but with a smaller $AUC_{FS} = 0.963$
			- `IF_RandomForest` → Here in the Score Plots Feature 1 is at the second place so we have a positive $AUC_{FS} = 0.451$ but we have the usual reduction in the precision values passing from 5 to 4 features for the `direct` approach and passing from 2 features to 1 in the `inverse` approach. 

### `moodify`

#### Importance Plots

- `EIF+`
	- `EXIFFI+, scenario 2`
		- Bar Plot: The `loudness` feature is the most important one around 60% of the times (probably in the first version of the paper there were some errors in the numerical order of the features and their names because this corresponds to Feature 3 (like the one we identified in the paper) but it is not `energy` as we wrote). In any case it makes sense also semantically that `loudness` is the most important feature to separate `energy` songs from `calm` songs. 
		- Score Plot: `loudness` is the first in the ranking with a clear advantage on `spec. rate` and `speechiness`. In any case all the features except `loudness` have really close importance values. 
	- `EXIFFI+, scenario 1`
		- Bar Plot: As usual in `scenario 1` there are several features appearing as the most important one in different runs.
		- Score Plot: Graphically we can see a gap in the importance scores of `speechiness`, `duration (ms)`, `liveness` and `spec_rate` with the respect to the others. `loudness` is at position 5. 
	- `RandomForest, scenario 2`
		- Bar Plot: `loudness` is at first place in all the runs.
		- Score Plot: `loudness` clearly on top, followed by `instrumentalness`, `speechiness`. `spec_rate` is at position 7. 
	- `RandomForest, scenario 1`
		- Bar Plot: `spec_rate`  is at the first place more than 50% of the times with `loudness` getting good percentages in position 2 and 3.
		- Score Plot: The first two features, with a visible margin over the others, are `spec_rate` and `liveness`. `loudness` is in position 7. 
- `EIF`
	- `EXIFFI, scenario 2`
		- Bar Plot: `loudness` is at first place almost in all the runs (probably there were 1/2 runs were `danceability` was at first place). 
		- Score Plot: `loudness` on top with a visible margin on `spec_rate`, `speechiness` and the other features. 
	- `EXIFFI, scenario 1`
		- Bar Plot: Similar to `EXIFFI+, scenario 1`
		- Score Plot: Like in `EXIFFI+, scenario 1` there are 4 features with a slight higher importance scores than the others. These 4 features are the same encountered in `EXIFFI+, scenario 1` but in a different order: `liveness,speechiness,spec_rate, duration(ms)`. `loudness` is at position 5. 
	- `RandomForest, scenario 2`
		- Bar Plot: Similar to `EIF+_RandomForest, scenario 2` with `loudness` at rank 1 in all the runs. 
		- Score Plot: Like in `EIF+_RandomForest, scenario 2` `loudness` is by far the most important feature, followed by `instrumentalness` and `speechiness`. 
	- `RandomForest, scenario 1`
		- Bar Plot: `speechiness` is at rank 1 about 90% of the times.
		- Score Plot: `speechiness` at first place followed by `liveness` and `loudness`. 
- `IF`
	- `DIFFI, scenario 2`
		- Bar Plot: A lot of variability in the first two ranks, `spec_rate` is the feature with the highest percentages.
		- Score Plot: `spec_rate` at first place followed by `duration(ms)` and `speechiness`. 
	- `DIFFI, scenario 1`
		- Bar Plot: Similar to `scenario 2`
		- Score Plot: Similar to `scenario 2`
	- `RandomForest, scenario 2`
		- Bar Plot: Here we have a perfect division between `loudness` and `instrumentalness` in the first two ranks (90% `loudness` and 10% `instrumnetalness` in rank 1 and viceversa in rank 2).
		- Score Plot: `loudness` at first place followed by `instrumentalness`. 
	- `RandomForest, scenario 1`
		- Bar Plot: Similar to `scenario 1` but now the first rank of the Bar Plot is divided by `spec_rate` and `instrumetalness`
		- Score Plot: `spec_rate` and `instrumentalness` at the first two places. 

#### Feature Selection Plots 

- `EIF+`
	- `scenario 2`
		- `EIF+`
			- `EXIFFI+` → $AUC_{FS} = 0.389$ not very good. There is an alternation between red and blue lines to be on top. In particular the blue line passes on top switching from 9 to 8 features (i.e. so when we remove `speechiness`) then the `direct` approach decreases in Average Precision when we pass from 6 to 4 features so when we remove `energy` and `instrumentalness`. On the other hand in the red line (i.e. `inverse` approach) it starts with a decreasing trend and then it start to increase from 6 to 4 features (i.e. when `energy` and `liveness` are removed), then it continues to increase. The higher jump is registered from 2 to 1 feature, this justifies the important of `loudness`. 
			- `EIF+_RandomForest` → $AUC_{FS} = 0.741$ sligthly better (probably because in the Score Plot there is a wider margin, in terms of importance, between `instrumentalness` and `speechiness`). In the `inverse` approach the Average Precision starts to increase from 4 features onward (i.e. in fact looking at the Score Plot the first 4 features have some margin on the others). In the `direct` approach instead there is a zig zag trend between 5 and 3 features.
				- Decrease between 6 and 5 → remove `danceability`
				- Increase between 5 and 4 → remove `spec_rate`
				- Decrease between 4 and 3 → remove `tempo`
			 This makes sense because we have seen `spec_rate` in high positions in the Score Plots of many configurations while `danceability` and `tempo` were never mentioned in the first positions. 
		- `EIF`
			- `EXIFFI` → $AUC_{FS} = 0.77$ Here we have the `inverse` approach always at higher precision than the `direct` one (as it should be). There is something like a U shape in `inverse` between 10 and 6 features. 
				- Decrease between 10 and 9 → remove `tempo`
				- Between 9 and 7 (i.e. remove `acousticness` and `danceability`) more or less constant
				- Increase between 7 and 6 → remove  `instrumentalness`
			 On the other hand in `direct` there is:
				- Increase between 5 and 4 → remove `instrumentalness`
				- Decrease between 4 and 2 → remove `danceability` and `acousticness`
			- `EIF_RandomForest` → $AUC_{FS} = 0.733$ More or less similar to `EXIFFI` but the $AUC_{FS}$ value is a bit lower because there is a point where the Average Precision of `direct` overcomes the `inverse` one. This point is at 5 features. The increase in precision going from 6 to 5 corresponds to the removal of `duration (ms)`. At the same time going from 6 to 5 a decrease is observed in the `inverse` approach, always for the removal of `duration (ms)`
		- `IF`
			- `DIFFI` → $AUC_{FS} = -1.037$ Negative value because `direct` is always at higher Average Precision values than `inverse`. In `direct` there is a peak in Average Precision that starts to decrease passing to 6 features, so when `loudness` is removed. The same happens in `inverse` where the precision drops passing from 5 to 4 features (always because of the removal of `loudness`). 
			- `IF_RandomForest` → $AUC_{FS} = 0.812$ Here in the Score Plot `loudness` is clearly the most important feature so now we still get a result that makes sense. Actually this is the highest $AUC_{FS}$ value found up to now. There is an increase in `direct` approach passing from 7 to 6 features (so removing `liveness`). At the same time when `liveness` is removed the increasing trend of the `inverse` approach starts. 

	- `scenario 1`
		- `EIF+`
			- `EXIFFI+` → $AUC_{FS} = -0.952$ Very similar to `IF_DIFFI, scenario 2`. In fact in both the Score Plots of these configurations `loudness` was placed at position 5. 
			- `EIF+_RandomForest` → In the Score Plot `loudness` is at position 7 so we have a negative $AUC_{FS} = -1.5$ and there is an Average Precision decrease going from 5 to 4 features (so when `loudness` is removed) in the `direct` approach. 
		- `EIF`
			- `EXIFFI` → $AUC_{FS} = -1.375$ Also here, like in `EXIFFI+, scenario 1` and `DIFFI, scenario 2`, feature `loudness` is at position 5 and so the final shape of the plot is similar to the ones already seen, although here the Average Precision values are a bit lower and so there is not a clear peak in the `direct` approach. 
			- `EIF_RandomForest` → $AUC_{FS} = -0.765$ In the Score Plot `loudness` is at the third rank so the $AUC_{FS}$ is negative and we have a decrease going from 9 to 8 features in `direct` approach and going from 3 to 2 in the `inverse` approach (i.e. when `loudness` is removed). 
		- `IF`
			- `DIFFI` → $AUC_{FS} = -1.019$ Same thing as `DIFFI, scenario 2`, `EXIFFI+/EXIFFI, scenario 1`
			- `IF_RandomForest` → Also here very similar to `scenario 1` (because, as it frequently happens, `loudness` is in $5^{th}$ place for importance values). However other than the usual decrease between 7 and 6 for `direct` and between 5 and 4 for `inverse` there are in `inverse` :
				- Increase between 3 and 2 → remove `liveness`
				- Decrease between 2 and 1 → remove `instrumentalness`

- `EIF`
	- `scenario 2`
		- `EIF+`
			- `EXIFFI+` → $AUC_{FS} = 0.353$ Very up and down trend. Let's analyse the main ups and down in both approaches. 
				- `direct`
					- Increase between 9 and 8 → remove `speechiness`
					- Increase between 7 and 6 → remove `liveness`
					- Decrease between 5 and 4 → remove `instrumentalness`
					- Increase between 2 and 1 → remove `tempo`
				- `inverse` → Here the trend start to be increasing (after a constant/sligthly decreasing trend) when we pass from 5 to 4 features → when `liveness` is removed. 
			- `EIF+_RandomForest` → $AUC_{FS} = 1.164$ Pretty standard Feature Selection plot. There are some ups and downs in the `direct` part. 
		- `EIF`
			- `EXIFFI` → $AUC_{FS} = 0.965$ Similar to `EIF+_RandomForest`. There is a V shape between 9 and 6 features in `inverse`: 
				- Decrease from 9 to 7 → remove `acousticness` and `instrumentalness`
				- Increase from 7 to 6 → remove `liveness`
			 On the other hand in `direct` we have: 
				 - Increase from 5 to 4 → remove `liveness`
				 - Decrease from 4 to 2 → remove `instruumentalness` and `acousticness`
				 - Increase from 2 to 1 → remove `tempo`
			- `EIF_RandomForest` → $AUC_{FS} = 0.91$ Between 6 and 4 features the `direct` approach is higher than the `inverse` one. V shape at the beginning in `direct`:
				- Decrease from 11 to 9 → remove `valence` and `acousticness`
				- Increase from 9 to 6 → remove `energy,tempo,danceability`
			On the other hand in `direct` the increasing trend starts from 4 features onward, we when `liveness` is removed. 
		- `IF`
			- `DIFFI` → As in `EIF+` here we are in the situation in which `loudness` is in position 5, so usual scenario → $AUC_{FS} = -1.22$
			- `IF_RandomForest` → $AUC_{FS} = 1.132$. Actually this is the highest $AUC_{FS}$ value seen up to now. It has the standard shape, with some oscillations in the `direct` approach. 

	- `scenario 1`
		- `EIF+`
			- `EXIFFI+` → Similar to `EIF+, EXIFFI+, scenario 1` → $AUC_{FS} = -1.642$
			- `EIF+_RandomForest` → Similar to `EIF+, EIF+_RandomForest, scenario 1` → $AUC_{FS} = -1.89$
		- `EIF`
			- `EXIFFI` → Similar to `EIF+, EXIFFI, scenario 1` → $AUC_{FS} = -1.83$
			- `EIF_RandomForest` → Similar to `EIF+, EIF_RandomForest, scenario 1` → $AUC_{FS} = -1.189$
		- `IF`
			- `DIFFI` → Similar to `EIF+, DIFFI, scenario 1` → $AUC_{FS} = -1.54$ 
			- `IF_RandomForest` → Similar to `EIF+, IF_RandomForest, scenario 1` → $AUC_{FS} = -0.844$

> [!note] 
> In `moodify` multiple times it happened that removing feature `liveness` increased a little bit the Average Precision. This suggests us that probably `liveness` is not a very important feature. 


> [!important] Particular Results in Synthetic Datasets
>  In most of the cases the Feature Selection plots done on `Xaxis` are almost full red and with very high values of $AUC_{FS}$ and so are pretty different from the ones I obtained in the first set of tests. The main difference is that now I am using `EIF+` to evaluate the Average Precision values while in the first version of the tests I always used the same model used for the interpretation to compute the Average Precisions (i.e. `EIF+` for `EXIFFI+`, `EIF` for `EXIFFI` and `IF` for `DIFFI`) . As we showed in the first version of the paper (in the Violin Plots) `EIF+` and `EIF` are very good with synthetic datasets (both in `scenario 1` and in `scenario 2`) while `IF` is very bad, in particular in `Xaxis` and `Yaxis` (while with the `bisect` datasets the performances are comparable with the ones of `EIF+` and `EIF`). 
>  We can see this effect mainly on the plot of `IF_RandomForest`. In the Score Plot of this configuration Feature 1 and Feature 0 have essentially the same importance but Feature 1 is sligthly above (and this does not follow the ground truth (that we know since we are working with a synthetic dataset) saying that Feature 0 is the most important feature). So we expect to have not a very good Feature Importance plot in terms of $AUC_{FS}$. In the first run of the tests in fact $AUC_{FS} = 1.523$ and there is a huge drop in precision (from about 0.7 to about 0.1) passing from 2 to 1 features (i.e. when Feature 0 is removed). However using `EIF+` as an evaluation model the result is $AUC_{FS} = 3.999$. We can see the fact that Feature 0 is wrongly at the second place because in the `direct` approach we have an increase in Average Precision passing from 6 to 5 features (i.e. when Feature 1 is removed) and in the `inverse` approach we have an even wider drop (from 0.9 to 0.1) passing from 2 to 1 features (i.e. when Feature 0 is removed). The reason why $AUC_{FS}$ is so high it's because in the `inverse` approach the Average Precision are all almost 1 while Feature 0 is part of the Feature Space. 

> [!warning] 
> The only doubt I have is that in configurations like `EIF+_EXIFFI+, scenario 1` and `EIF+_RandomForest, scenario 1` (that were evaluated with `EIF+` also in the first set of tests) the plots are very different. In the first set of tests in fact these plots were "half red" and the shape was more or less the one of a rectangular triangle (triangolo rettangolo), but now, in this second run of tests, they are full red with a rectangular shape like most of the ones done up to now. 
> Looking at all the Score Plots of these configurations there is always Feature 0 on top, the difference is the margin from the importance score of Feature 1 (that in some cases is high and some other is smaller). So essentially it seems that while Feature 0 is in the mix (and so in all the steps for the `inverse` approach if it is the first feature) the Average Precision (in `Xaxis`, and probably also on `Yaxis`) is always very close to 1. 

## TO DO today 12/3

- [ ] Finish the re doing experiments on `Xaxis` without pre processing the data 
	- [ ] Feature Selection Plots
	- [ ] Local Scoremaps
- [ ] Re do the experiments also on `bisect_3d` without pre processing the data 
	- [ ] Global Importances
	- [ ] Feature Selection Plots
	- [ ] Contamination Plots
	- [ ] Local Scoremaps
- [ ] Do the Performance Report experiments on `bisect_3d` (DO NOT PRE PROCESS THE DATA)



