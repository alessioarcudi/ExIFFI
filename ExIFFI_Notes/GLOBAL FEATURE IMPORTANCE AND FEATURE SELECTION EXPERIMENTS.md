
# Experiment Configurations

> [!note] Hyper parameters
>  - `models = [IF,EIF,EIF+]`
>  - `interpretation = [EXIFFI,DIFFI,RandomForest]`
>  - `scenario=[1,2]`

- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF+` , `EXIFFI` and `scenario=2`|`EIF+` , `EXIFFI` and `scenario=2`]] → ==completed==
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF+` , `EXIFFI` and `scenario=1`|`EIF+` , `EXIFFI` and `scenario=1`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF` , `EXIFFI` and `scenario=2`|`EIF` , `EXIFFI` and `scenario=2`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF` , `EXIFFI` and `scenario=1`|`EIF` , `EXIFFI` and `scenario=1`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `DIFFI` and `scenario=2`|`IF` , `DIFFI` and `scenario=2`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `DIFFI` and `scenario=1`|`IF` , `DIFFI` and `scenario=1`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF+` , `RandomForest` and `scenario=2`|`EIF+` , `RandomForest` and `scenario=2`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF+` , `RandomForest` and `scenario=1`|`EIF+` , `RandomForest` and `scenario=1`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF` , `RandomForest` and `scenario=2`|`EIF` , `RandomForest` and `scenario=2`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`EIF` , `RandomForest` and `scenario=1`|`EIF` , `RandomForest` and `scenario=1`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `RandomForest` and `scenario=2`|`IF` , `RandomForest` and `scenario=2`]]
- [[GLOBAL FEATURE IMPORTANCE AND FEATURE SELECTION EXPERIMENTS#`IF` , `RandomForest` and `scenario=1`|`IF` , `RandomForest` and `scenario=1`]]

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

##  `EIF+` , `EXIFFI` and `scenario=2`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok

#### Real World Dataset

- `wine` → Re do after having added `feature_names`
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breastw` → ok
- `ionosphere` → ok
- `annthyroid`  → Re do after having added `feature_names`
- `pendigits` → ok
- `diabetes` → ok
- `shuttle` → ok
- `moodify` → ok

## `EIF+` , `EXIFFI` and `scenario=1`

### Synthetic Dataset 

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →

### Real World Dataset 

- `wine`  →
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 
	- Global Importance → Job 201405
	- Feature Selection → Job 201408
## `EIF` , `EXIFFI` and `scenario=2`

#### Synthetic Datasets

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →

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
- `moodify` → 
	- Global Importance → Job 201403
	- Feature Selection → Job 201409

## `EIF` , `EXIFFI` and `scenario=1`

### Synthetic Datasets

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  →
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 
	- Global Importance → Job 201404
	- Feature Selection → Job 201410
## `IF` , `DIFFI` and `scenario=2`

### Real World Datasets

- `wine` → On CAPRI 
- `moodify` → Job 201261 on CAPRI

## `IF` , `DIFFI` and `scenario=1`

### Synthetic Datasets

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  →
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 

## `EIF+` , `RandomForest` and `scenario=2`

### Synthetic Datasets

- `Xaxis` → 
- `Yaxis` → 
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  → ok 
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 

## `EIF+` , `RandomForest` and `scenario=1`

### Synthetic Datasets

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  →
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 

## `EIF` , `RandomForest` and `scenario=2`

### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  → ok
- `glass` → 
- `cardio` →
- `pima` → ok, relocate the $AUC_{FS}$ box 
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 

## `EIF` , `RandomForest` and `scenario=1`

### Synthetic Datasets

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  →
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 

## `IF` , `RandomForest` and `scenario=2`

### Synthetic Datasets

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  →
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 

## `IF` , `RandomForest` and `scenario=1`

### Synthetic Datasets

- `Xaxis` →
- `Yaxis` →
- `bisect` →
- `bisect_3d` →
- `bisect_6d` →
### Real World Datasets

- `wine`  →
- `glass` →
- `cardio` →
- `pima` →
- `breast`
- `ionosphere`
- `annthyroid` →
- `pendigits` →
- `diabetes`→
- `shuttle`
- `moodify` → 

## Problem with `pima,ionosphere` and `breastw` datasets

`pima`, `ionosphere` and `breastw` have an high contamination factor: 34.89%, 35.71% and 52.56% respectively. In particular `train_size`=0.8 + these contamination percentanges is higher than 1.

In the case of `pima` we call `dataset.split_dataset(train_size=0.8,contamination=0)` in Scenario 2. So essentially we want to create a training set of only inliers taking a subsample containing 80% of the inliers of the original dataset.

> [!example] `pima`
> `pima` has 768 samples, so 80% of the inliers is 614.4. We round this number to 614.

> [!important] 
> The problem is that if we divide `dataset.y` into the inliers and outliers indexes we have 268 outliers and 500 inliers. So we can't take 614 inliers from the original dataset. 

> [!example] `ionosphere`
> In case of `ionosphere` the inliers are 225 but with `train_size=08` we try to take 280 inliers from the original dataset. 

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


