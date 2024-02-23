
# Experiment Configurations

- `models = [IF,EIF,EIF+]`
- `interpretation = [EXIFFI,DIFFI,RandomForest]`
- `scenario=[1,2]`

## `EIF+` , `EXIFFI` and `scenario=2`

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
	- Global Importance 40 run pre process → `22-02-2024_22-56` → ok
	- Feature Selection → `22-02-2024_22-50` → ok

### Real World Dataset

- `wine` :
	- Global Importance 10 run → `22-02-2024_21-38`
	- Feature Selection → `22-02-2024_21-53` → no sense → maybe it's the dataset that is bad 
	- Global Importance 40 run → `22-02-2024_22-02`
	- Feature Importance → `22-02-2024_22-07`
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

## `EIF+` , `EXIFFI` and `scenario=1`

- `wine` :
	- Global Importance 10 run → `22-02-2024_22-14`
	- Feature Selection → `22-02-2024_22-16` → no sense 

## `EIF` , `EXIFFI` and `scenario=2`

- `wine` :
	- Global Importance 40 run pre process → `22-02-2024_22-37` → with pre process much better than without pre process
	- Feature Selection → `22-02-2024_22-38` → no sense 

