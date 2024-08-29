Let's use this note to keep track of the experiments I am running for the resubmission of `ExIFFI` after the review from `EAAI`. 

# Experiments to perfom

- Time Scaling Experiments for `KernelSHAP`
- Experiments for the `ECOD` AD model
- Correlation between Feature Importance and Anomaly Score 

# Time Scaling `KernelSHAP`

- Datasets varying samples and with 6 features
	- `Xaxis_100_6` → ==ok==
	- `Xaxis_250_6` → ==ok==
	- `Xaxis_500_6` → ==ok==
	- `Xaxis_1000_6` → ==ok==
	- `Xaxis_2500_6` → ==ok==
	- `Xaxis_5000_6` → ==ok==
	- `Xaxis_10000_6` → ==ok==
	- `Xaxis_25000_6`  → ==ok==
	- `Xaxis_50000_6`  → running 
	- `Xaxis_100000_6` → running 
	- `Xaxis_25000_6`
	- `Xaxis_300000_6`
- Datasets varying number of features and with 5000 samples
	- `Xaxis_5000_6`
	- `Xaxis_5000_16`
	- `Xaxis_5000_32`
	- `Xaxis_5000_64`
	- `Xaxis_5000_128`
	- `Xaxis_5000_1024`
	- `Xaxis_5000_4096`

# Experiments `ECOD` model 

Use the `PyOD` implementation of [ECOD](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.ecod)

# Experiments Correlation

> [!todo] 
> Do this procedure for each interpretation method we compared inside the paper. 
> - Do a table with the correlation values → add a column to the tables of the Average Precision,Time, 
> - For each dataset
> 	- Compute the `LFI` of all the samples 
> 	- Compute the sum of the `LFI` scores over all the features
> 	- Compute the Anomaly Score of the AD method 
> 	- Compute the correlation 

^c65818
