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