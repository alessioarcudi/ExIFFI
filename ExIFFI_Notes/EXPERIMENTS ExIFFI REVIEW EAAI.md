Let's use this note to keep track of the experiments I am running for the resubmission of `ExIFFI` after the review from `EAAI`. 
# Experiments to perform

- Time Scaling Experiments for `KernelSHAP`
- Experiments for the `ECOD` AD model
- Correlation between Feature Importance and Anomaly Score 
- New synthetic dataset experiment
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
	- `Xaxis_50000_6`  → ==ok==
	- `Xaxis_100000_6` → running 
- Datasets varying number of features and with 5000 samples
	- `Xaxis_5000_16` → ==ok==
	- `Xaxis_5000_32` → ==ok== 
	- `Xaxis_5000_64` → running 
	- `Xaxis_5000_128` → 
	- `Xaxis_5000_256` → 
	- `Xaxis_5000_512` → 

# Experiments `ECOD` model 

Use the `PyOD` implementation of [ECOD](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.ecod). The `ECOD` model has some sort of interpretation but it is only local and only a graphical interpretation if I do remember correctly thus there should not be something like the `LFI` or `GFI` score to exploit for a comparison with `ExIFFI`. So we can use it as we used `DIF` and `Autoencoder`. So we will use it in the experiments:

- Contamination Experiment
- Precision metric experiment 
- Time Scaling `fit-predict` experiment 

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

# New synthetic dataset experiment

The idea of this experiment is to create a new type of synthetic dataset. This dataset is similar to `Bisect3D` but instead of generating the three anomalous features using the same $[\text{min},\text{max}]$ interval for the Uniform distribution $\mathcal{U}_3$ we use three overlapping intervals of different lenghts for example:

- $[2,4]$ for Feature 0
- $[2,6]$ for Feature 1
- $[2,8]$ for Feature 2

In this way we should have the three anomalous features on top of the `GFI` ranking but not all with similar importance values but we should have Feature 2 on top followed by Feature 1 and Feature 0 because the `anomaly_interval` of Feature 2 is wider than the ones of Feature 1 and Feature 0 and thus the anomalies along that feature are more isolated. 
