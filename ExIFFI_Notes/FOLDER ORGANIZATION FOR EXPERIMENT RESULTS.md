# Folder Organization

- `ExIFFI`
	- `experiments`
		- `results` → One folder per dataset 
			- `wine` 
			- `glass`
			- `pima`
			- `breastw`
			- `ionosphere`
			- `cardio`
			- `annthyroid`
			- `pendigits`
			- `diabetes`
			- `shuttle`
			- `moodify`
				- `experiments`
					- `global_importances` → Results of experiments done with `compute_global_importances`
						- `ExIFFI` → Here we insert different importances matrices obtained from different experiments
							- `imp_mat_1`
							- `imp_mat_2`
							- ...
						- `DIFFI` → Here we insert different importances matrices obtained from different experiments
							- `imp_mat_1`
							- `imp_mat_2`
							- ... 
					- `metrics` → Here we save a `pd.DataFrame` with a single row containing all the parameter values per experiment 
						- `EIF+`
						- `IF`
						- `EIF`
						- `AutoEncoder`
						- `DIF`
				- `plots` → Here we save the different plots obtained from the experiments (e.g. Bar Plot, Score Plot, Importance Map,...)
					
						

## Function `performance` in `utils`

Create a function to save the experiments results. It takes in input: 


- `y_pred`
- y
- model
- model name 
- dataset 
- dataset name 
- Contamination 
- Train Fraction 
- Return Dataframe with all the columns specified above 
- Column with TimeStamp 
- Results will go in →  `/experiments/results/wine/experiments/metrics/EIF+/filename `  
- `filename` just with the timestamp. 

## `split`

`dataset.split_dataset()` returns: 

- `dataset.X_train` → A portion of `dataset.X` (the portion is specified by the parameter `train_size` (e.g. in `wine` we have 129 samples if I pass `train_size=0.8` I end up with 103 samples)). The number of outliers to insert in this portion of the dataset is specified by the input parameter `contamination`
- `dataset.y_train` → Allineate the labels contained in `y` to the samples extracted in `X_train`

So if I have `split=True`:

- Train on `dataset.X_train` (the labels are `dataset.y_train`)
- Test on `dataset.X` (the labels are `dataset.y`)
- `pre_process` function
	- Normalize `dataset.X_train`
	- Normalize `dataset.X` as it is the test set → no need to do `np.r_[X_train,X_test]`
	- So pass `pre_process(dataset.X_train,dataset.X,split=True)`

If I have `split=False`:

- Train and test on `dataset.X` (labels are `dataset.y_train`)
- `pre_process` function
	- Normalize `dataset.X_train`
	- So pass `pre_process(dataset.X,dataset.X,split=False)`