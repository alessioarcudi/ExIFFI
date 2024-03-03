
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

## TODO as soon as possible 

- [x] Redo the feature selection plots using the new `aucfs` values (also the negative ones)
- [x] Re do the feature selection plot for `annthyroid, EIF+, RandomForest, scenario=1,2` (apparently they have disappeared)
- [x] When Alessio inserts the time computation in the experiment scripts re run all the experiment to take the time into account. In particular re run the configurations `IF, DIFFI` and `IF, RandomForest` since they where wrong in the implementation used up to now. 
- [x] Try to use `dataset.downsample` on `diabetes, shuttle, moodify` to see how the results change with the respect to the experiment results on the entire dataset → they essentially stay the same.

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
		- Bar/Score Plots: Here the dominance of Feature 0 is even more evident in the Score Plot (the Bar Plot is more or less the same as in `EXIFFI, scenario=2`). That's because we use the `EIF+` scores as labels for a `RandomForest` regressor that than has a probably more robust way to compute the Feature Importances. Obviously the problem of `RandomForest` is that it needs to labels.
		- Feature Selection Plot: Very similar to `EXIFFI, scenario=2` → $AUC_{FS} = 4.539$
	- `RandomForest, scenario=1`
		- Bar/Score Plots: Also in this case in the Score Plot Feature 0 is the most important but with a lower margin on the others while in the Bar Plot it is still at the first position in all the 40 runs
		- Feature Selection Plot: Like for `EXIFFI, scenario=1` the area between the red and blue line reduces to more or less half of the previous value → $AUC_{FS} = 2.92$
- `EIF`
	- `EXIFFI, scenario=2`
		- Bar/Score Plots: Similar to `EXIFFI, scenario=2` with a clear dominance of Feature 0 on the others
		- Feature Selection Plot: Similar to `EIF+, EXIFFI, scenario=2` with even a sligthly better $AUC_{FS}$ value → $AUC_{FS} = 4.733$. That is probably because we have still a single anomalous feature. `EIF+` should be better in cases of multiple "anomalous" features
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
	- `DIFFI, scenario=2` → ==review after the new experiments==
		- Bar/Score Plots: Very bad. The importance scores are assigned more or less randomly. Probably if we launch another time the experiment the feature order may change. The Score Plot is not very informative because all the importance values are very similar and there is not a feature that raises with the respect to the others. This probably happens because in scenario 2 we train only on the inliers and the `IF` model probably needs to see some outliers in order to work well. 
		- Feature Selection Plots: The blue and red line are overlapped for all the points except for 3 features where the red one is sligthly higher. As a consequence the $AUC_{FS}$ metric is very low → $AUC_{FS} = 0.037$
      - `DIFFI, scenario=1`  → ==review after the new experiments==
		  - Bar/Score Plots: In scenario 1 the results are much better and resemble the typical results seen in scenario 2 for the previous configurations with Feature 0 raising as the most important feature. This result confirms the fact that `IF` requires to have some contamination in the training set to work well. So also for the rest of the datasets we should expect to have better results in `scenario=1`  for the `IF` model. 
		  - Feature Selection Plot: Here the results are worse in the sense that the blue line is always higher than the red one and this causes the $AUC_{FS}$ metric to be negative → $AUC_{FS} = -0.19$
	 - `RandomForest, scenario=2` 
		 - Bar/Score Plots: As for `IF, DIFFI, scenario=2` the `IF` model does not work well in scenario 2 and so the results are pretty random, this can be noticed also looking at the error bars reported in the Score Plot (to represent the standard deviation) that are very long compared to the importance values. **After re run** → the results are still quite random but Feature 0 is the most important in almost 60% of the cases in the Bar Plot and it is first in the Score Plot (not by a lot but still first). 
		 - Feature Selection Plot: Very bad as expected → $AUC_{FS} = -0.139$. **After re run** → The Feature Selection plot is much better → $AUC_{FS} = 2.55$
	 - `RandomForest, scenario=1` 
		 - Bar/Score Plots: In this case passing to scenario 1 does not help `IF` to achieve better results. Strangely Feature 0 is even considered the least important feature with a small margin on the others 
		 - Feature Selection Plot: $AUC_{FS} = 0.188$

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
	- `DIFFI, scenario=2` → ==review after the new experiments==
		- Bar/Score Plots: Very random as in `Xaxis`
		- Feature Selection Plots: Bad as in `Xaxis` → $AUC_{FS} = -0.195$
	- `DIFFI, scenario=1` → ==review after the new experiments==
		- Bar/Score Plots: As in `Xaxis` with scenario 1 the situation is better and Feature 1 is clearly the most important one
		- Feature Selection Plots: As in `Xaxis` not good → $AUC_{FS} = -0.191$
	- `IF, scenario=2` 
		- Bar/Score Plots: Similarly to `Xaxis` the results are more or less random but with a sligth advantage of Feature 1. 
		- Feature Selection Plots: Similar to `Xaxis` → $AUC_{FS} = 2.494$
	- `IF, scenario=1` → re do the experiments 

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
	- `DIFFI, scenario=2` → ==review after the new experiments==
	- `DIFFI, scenario=1` → ==review after the new experiments==
	- `RandomForest, scenario=2`
		- Bar/Score Plots: Similar to `EIF, RandomForest, scenario=2` with a significant different in the importances of Feature 0 and Feature 1
		- Feature Selection Plot: $AUC_{FS} = 4.466$ → Very high → also higher than `EIF+, EXIFFI, scenario=2`
	- `RandomForest, scenario=1` → ==review after the new experiments==

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
	- `DIFFI, scenario=2` → ==review after the new experiments==
	- `DIFFI, scenario=1` → ==review after the new experiments==
	- `RandomForest, scenario=2`: 
		- Bar/Score Plots: As in `EIF+, RandomForest, scenario=2` Feature 2 is the most important
		- Feature Selection Plot: $AUC_{FS} = 3.996$ → very high value. All the precision values in the `inverse` red line are really close to 1. 
	- `RandomForest, scenario=1` → ==review after the new experiments==

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
	- `DIFFI, scenario=2` → ==review after the new experiments==
	- `DIFFI, scenario=1` → ==review after the new experiments==
	- `RandomForest, scenario=2`
		- Bar/Score Plots: `RandomForest` finds Feature 2 as the most important as usual → 3 and 0 are the second a third most important
		- Feature Selection Plot: $AUC_{FS} = 0.235$

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
