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
	- `Xaxis_5000_64` → ==ok== 
	- `Xaxis_5000_128` → 
	- `Xaxis_5000_256` → 
	- `Xaxis_5000_512` → 
# Experiments `ECOD` model 

Use the `PyOD` implementation of [ECOD](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.ecod). The `ECOD` model has some sort of interpretation but it is only local and only a graphical interpretation if I do remember correctly thus there should not be something like the `LFI` or `GFI` score to exploit for a comparison with `ExIFFI`. So we can use it as we used `DIF` and `Autoencoder`. So we will use it in the experiments:

- [x] Precision metric experiment
- [x] Contamination experiment
- [ ] Time Scaling `fit-predict` experiment 

## `ECOD` Performances

Looking at the results obtained with the Contamination plots we can see how `ECOD` has similar performances to the other model only on the `bisect` datasets while on the others the performances are not very good. A possible explanation for this behavior may be connected to the fact that `ECOD` makes the assumption that the **features in the dataset are independent**. In fact it computes the Anomaly Scores individually on each feature (using the `ECDF` function estimates on each single feature) and then combines all the univariate scores to obtain a single score for each sample. The problem is that this assumption likely does not hold on the benchmark datasets we used in this paper. This is certified by the fact that in most cases there are multiple features with high importance values. On the other hand in the synthetic datasets we can safely say that the features are independent (because we generated each one of them sampling from a certain distribution) and in fact in the `bisect` datasets the model works decently. The strange thing are probably the bad performances on `Xaxis` and `Yaxis`. 
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

The idea of this experiment is to create a new type of synthetic dataset. This dataset is similar to `Bisect3D` but instead of generating the three anomalous features using the same $[\text{min},\text{max}]$ interval for the Uniform distribution $\mathcal{U}_3$ we use three overlapping intervals of different lengths for example:

- $[2,4]$ for Feature 0
- $[2,6]$ for Feature 1
- $[2,8]$ for Feature 2

In this way we should have the three anomalous features on top of the `GFI` ranking but not all with similar importance values but we should have Feature 2 on top followed by Feature 1 and Feature 0 because the `anomaly_interval` of Feature 2 is wider than the ones of Feature 1 and Feature 0 and thus the anomalies along that feature are more isolated. 

## Experiments to perform on `bisect_3d_prop`

This is the complete list of experiments we do in general on any dataset:

- `global_importances` → For each interpretation method compute the `GFI` scores and produce the Bar and Score Plot. In any case then in the paper we just report the Bar and Score Plot of `EXIFFI+` so we can just stick to this experiment 
- `local_importances` → Obtain the Local Scoremaps for each interpretation algorithm → also in this case we report just the `EXIFFI+` ones
- `ablationEIF+` → Ablation study on the `EIF+` model were we try to see how the Average Precision changes as we change the hyperparameter $\eta$ 
- Contamination Experiment → This one has to be done for each one for of the AD models under comparison (i.e. `Autoencoder`,`DIF`,`ECOD`,`EIF`,`EIF+`,`IF`) and then we have to use the `plot_contaminations.py` script to merge all the experiments into a single plot
- Feature Selection Experiment → Here we have to do the Feature Selection experiment for all the 24 combinations, even though at the end we report just 6 of them in the final paper (all evaluated using `EIF+` as the `AD` model) 
	- `EIF+_EXIFFI+`
	- `EIF+_EXIFFI`
	- `EIF+_IF_EXIFFI`
	- `EIF+_EIF+_RandomForest`
	- `EIF+_EIF+_RandomForest`
	- `EIF+_IF_RandomForest`
- Metrics Experiment → Similarly to the Contamination Experiment this has to be done for each one of the `AD` models 
- Time Scaling Experiment → This one is needed to put execution times on the metrics table for the `fit` and `predict` and also in the $AUC_{FS}$ table (the importance time in this case). Although in this case we can also copy the times we put for `bisect_3d` (in fact the shape of the two datasets are identical so also the execution times should be very similar) 

- [x] `GFI` experiment for `EXIFFI+`
- [x] Local Scoremaps `EXIFFI+`
- [ ] `ablation_EIF+` experiment 
- [x] Contamination Experiment 
- [ ] Feature Selection Experiment
- [x] Metrics Experiment
- [ ] Time Scaling Experiment 