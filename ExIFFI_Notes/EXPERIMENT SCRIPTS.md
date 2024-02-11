Let's draw a scheme on how to set and organize the experiments to do on the paper. 

We need two Python Script for two different kind of experiments: 

1. ExIFFI Experiment → Experiment to launch on each different benchmark dataset that returns:
	- Bar Plot 
	- Score Plot
	- Importance Map 
	- Complete Importance Map 
	- Metrics  $AUC\tilde{S}_{top}$ and $F1\tilde{S}$ to evaluate the interpretation model ? 
1. Model Comparison Experiment → Experiment to launch on each different benchmark dataset and on each different Anomaly Detection model to create a comparison in terms of performances. For each model we want to return the following metrics: 
	- Average Precision 
	- ROC AUC Score 
	- Precision 
	- Recall
	- `F1` Score 
	- Accuracy 
	- Balanced Accuracy 

# ExIFFI Experiment 

We can use these experiments to conduct the Ablation Studies on ExIFFI: 

- Change the contamination factor in the training set → In the DIF paper they used 11 contamination factors between 0 and 10%. So we can do something like `np.linspace(0,10,11)` → so $[0,1,2,3,\dots,10]$
	- Use a different percentage to divide into inliers and outliers to compute GFI in the `Global_Importance` function → use the same set of contamination values reported above 

> [!note] 
> These two points can be considered together → if we use a contamination factor of 0.01 we will divide inliers and outliers using a percentage of 0.01  

- Use a different number of trees in the definition of the ExIFFI model: 
	- 100 
	- 300
	- 600
- Include or exclude the `depth_based` parameter 
- Different number of runs for the GFI Computation in `compute_global_importance` method 
- Use synthetic datasets, as done in the DIF paper, to see how the model scales to dataset of increasing size and increasing dimensionality. Use the usual synthetic dataset and perform experiments computing the execution time: 
	- Keeping constant the sample size and increasing the dimensionality (e.g. `np.linspace(16,4096,9)`)
	- Keeping constant the dimensionality and increasing the sample size (e.g. `np.linspace(1000,256000,9)`) 

> [!note] 
> For problems related to the fact that`plt.show()` blocks the execution of the script it is not possible to plot the Complete Scoremap with one execution of `test_exiffi.py`. We are in any case able to produce the Bar Plot, Score Plot and Importance Map plots. The Complete Scoremap (which in any case are not essential for the paper) can be produced separately in another script. 
## Plots to produce 

- Compute the  $AUC\tilde{S}_{top}$ and $F1\tilde{S}$ metrics for different experiment configurations and produce a plot to see how they vary with the respect to:
	- Contamination factor
	- Number of trees
	- `depth_based` parameter
- Do a subplot putting together the Bar Plot/Score Map/Importance Scoremap obtained with different configurations to see what changes in the interpretation 
- Plot `sample_size` vs `execution_time`
- Plot `n_features` vs `execution_time`

# Model Comparison Experiment 

Anomaly Detection models to compare: 

- Isolation Forest (IF)
- Extended Isolation Forest (EIF)
- Extended Isolation Forest Plus (EIF+)
- Deep Isolation Forest (DIF)
- AutoEncoder (AE)

- Compare the different models on the Classification metrics producing a Table like Table 5 of the DIF paper 
- Compare the different models on their scaling abilities → use the synthetic dataset with different `sample_size` and `n_features` as done in [[EXPERIMENT SCRIPTS#ExIFFI Experiment|ExIFFI Experiment]]

We can also perform an Ablation Study on EIF+: 

- Change the contamination factor in the training set
- Different number of trees
- Use different Scalers to pre-process the data:
	- `StandardScaler`
	- `MinMaxScaler`
	- `MaxAbsScaler`
	- `RobustScaler`
- Use different distributions to sample the intercept point `p` of the EIF+ separating hyperplanes:
	- $N(mean(X),\eta \ std(X)$ changing the value of $\eta$ 
	- $N(median(X),\eta \ std(X)$ changing the value of $\eta$ 
	- $U(\frac{min(X)}{\lambda},\lambda \ max(X))$ with $\lambda > 1$

> [!note] Sampling `p` from  $U(\frac{min(X)}{\lambda},\lambda \ max(X))$ with $\lambda > 1$
>  This is a modified version of the distribution used in EIF but dividing by $\lambda$ the lower bound and multiplying by $\lambda$ the upper bound we are enlarging the interval of possible values so that there is also the possibility to create cuts surrounding the training set distribution. However since the distribution is uniform there is also the possibility of sampling values of `p` inside $(min(X),max(X))$ like the cuts done by EIF. 



