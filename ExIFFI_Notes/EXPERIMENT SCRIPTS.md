Let's draw a scheme on how to set and organize the experiments to do on the paper. 

We need two Python Script for two different kind of experiments: 

1. ExIFFI Experiment → Experiment to launch on each different benchmark dataset that returns:
	- Bar Plot 
	- Score Plot
	- Importance Map 
	- Complete Importance Map 
	- Metrics  $AUC\tilde{S}_{top}$ and $F1\tilde{S}$ to evaluate the interpretation model ? 
2. Model Comparison Experiment → Experiment to launch on each different benchmark dataset and on each different Anomaly Detection model to create a comparison in terms of performances. For each model we want to return the following metrics: 
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
> For problems related to the fact that `plt.show()` blocks the execution of the script it is not possible to plot the Complete Scoremap with one execution of `test_exiffi.py`. We are in any case able to produce the Bar Plot, Score Plot and Importance Map plots. The Complete Scoremap (which in any case are not essential for the paper) can be produced separately in another script. 
## Plots to produce 

- Compute the  $AUC\tilde{S}_{top}$ and $F1\tilde{S}$ metrics for different experiment configurations and produce a plot to see how they vary with the respect to:
	- Contamination factor

> [!note] 
> In order to perform experiments changing the contamination factor I added the parameter `contamination` to classes `ExtendedIF` and `Extended_DIFFI_parallel`. This addition is helpful also because it let us remove the parameter `p` from the methods `predict`  and  `evaluate` of `ExtendedIF`. I also substituted the factor `0.1` in method `Global_importance` of `Extended_DIFFI_parallel` with `self.contamination`. So when we change the contamination factor we also change the percentage of samples considered as outliers in the computation of the GFI Score. 

- Number of trees 
- `depth_based` parameter 

- Do a subplot putting together the Bar Plot/Score Map/Importance Scoremap obtained with different configurations to see what changes in the interpretation 
- Plot `sample_size` vs `execution_time`
- Plot `n_features` vs `execution_time`

### To do 

- [ ] Integrate the changes done in the code into the code written in `numba` by Alessio 
- [ ] Work with the `wrapper` and the `add_bash` commands to perform the experiments similarly to what was done for the HPC Project 
	- [ ] The I can try to execute the experiments in the same way done in CAPRI (so using a `sh` script the launches successively the command to execute the script for each different dataset separately) locally on my PC (I do not think it will require so much time (also if I use the `numba`/C optimized))
- [ ] Try to create a subplot with the plots produced by `bar_plot`, `score_plot` and `importance_map`

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

> [!note] 
> Also here I added the `contamination` parameter to class `ExtendedIsolationForest` and, as a consequence, `IsolationForest`. In this way it is possible to remove parameter `p` from the `predict` method. 
> The default value for `contamination` is `auto` that corresponds to a contamination factor of `0.1`. The contamination value can also be set by the user passing a `float` but it cannot be higher than `0.5`. 

- Different number of trees
- Use different Scalers to pre-process the data:
	- `StandardScaler`
	- `MinMaxScaler`
	- `MaxAbsScaler`
	- `RobustScaler`
- Use different distributions to sample the intercept point `p` of the EIF+ separating hyperplanes:
	- $N(mean(X),\eta \ std(X)$ changing the value of $\eta$ 
	- $N(median(X),\eta \ std(X)$ changing the value of $\eta$ 
	- $U(\frac{min(X)}{\eta},\eta \ max(X))$ with $\eta > 1$ ^418327
- Use different values of $\eta$

> [!note] Sampling `p` from  $U(\frac{min(X)}{\lambda},\lambda \ max(X))$ with $\lambda > 1$
>  This is a modified version of the distribution used in EIF but dividing by $\lambda$ the lower bound and multiplying by $\lambda$ the upper bound we are enlarging the interval of possible values so that there is also the possibility to create cuts surrounding the training set distribution. However since the distribution is uniform there is also the possibility of sampling values of `p` inside $(min(X),max(X))$ like the cuts done by EIF. 

^dbaab1

### To do 

- [ ] Find some other function to use instead of `std` to define the distributions because the `std` takes a lot to be computed. 
	- [ ] Write a separate function that computes the `std` applying the definition (i.e. average of the squared differences of each point from the mean) and compile it with `numba`
	- [ ] Do the same writing the `std` in C and use `ctypes` to invoke the function in Python

> [!note] 
> Scusa Alessio ma sei sicuro che il calcolo della std sia lento?. Perchè io ho provato a fare una funzione con numba e con C per velocizzare il calcolo della std ma:
> 
> - Con numba è veloce uguale
> - Con C è addirittura più lenta
> 
> Numpy è scritto e ottimizzato in C quindi non credo si possa fare più veloce di `np.std`  

#### Metrics similar to `std`

- `variance` → It is slightly faster to compute → it's like the `std` but we do not have to do `sqrt` at the end
- `range` → Difference between maximum and minimum values in a dataset → it's sensible to extreme values/outliers. This may create problems if we include some anomalies in the training set. It is just the difference between the max and the min → its complexity depends on how complex is to compute the max and the min. → It is slightly faster than `std`



On `moodify`  normalized → 1 order of magnitude faster

``` 
std Time: 0.0010101795196533203 
std: 1.0658973576087336 
################################################## 
range Time: 0.0003421306610107422 
range: 19.634186249839274 
################################################## 
Time difference: 0.0006680488586425781
```

However the value is pretty different and much higher → we may end up doing cuts that are very far from our cluster of normal points. Maybe if we want to use `range` we can divide by the parameter $\eta$ instead of multiplying by it. 

- `Interquartile Range(IQR)` → $IQR=Q1-Q3$ → less sensitive to outliers with the respect to `range`. It is similar to `range` so it may be a little faster to compute with the respect to `std`

On `moodify` normalized 

``` 
std Time: 0.0012149810791015625 
std: 1.3360755010639553 
################################################## 
IQR Time: 0.00673365592956543 
IQR: 1.7766700729898175 
################################################## 
Time difference: -0.005518674850463867
```

The values are pretty similar but unfortunately the IQR is sligthly slower than `std`. 

- `Mean Absolute Deviation (MAD)` → Average of the absolute differences between each data point and the mean.

On `moodify` normalized → 1 order of magnitude better

``` 
std Time: 0.0011882781982421875 
std: 1.2166114762203248 
################################################## 
MAD Time: 0.0007545948028564453 
MAD: 0.8354337822996187 
################################################## 
Time difference: 0.0004336833953857422
```

The value of MAD here is also not so different from the `std` one so this metric may be a good choice. 

##### `normal_range`

- `normal_mean_range`

$$
	N(mean(X),range(X))
$$
- `normal_mean_range_eta`

$$
	N(mean(X),\frac{range(X)}{\eta})
$$

- `normal_median_range`

$$
	N(median(X),range(X))
$$
- `normal_median_range_eta`

$$
	N(median(X),\frac{range(X)}{\eta})
$$

##### `normal_MAD`

- `normal_mean_MAD`

$$
	N(mean(X),MAD(X))
$$
- `normal_mean_MAD_eta`

$$
	N(mean(X),\frac{MAD(X)}{\eta})
$$

- `normal_median_MAD`

$$
	N(median(X),MAD(X))
$$
- `normal_median_MAD_eta`

$$
	N(median(X),\frac{MAD(X)}{\eta})
$$

## Parameters `distribution` and `eta`

The `distribution` and `eta` parameters were added in classes `ExtendedIsolationForest` and `Extended_DIFFI_parallel` so that it is possible to do experiments on the EIF+ and ExIFFI models **changing the distribution from which the intercept point `p` is sampled** and **changing the scaling factor $\eta$ of this distribution**.

### `distribution`

The `distribution` parameter is passed as a string and can take the following three values: 

- `normal_mean`: This is the default value and the distribution we have used up to now for the EIF+ model: 

$$
	N(mean(X),\eta \ std(X)
$$

^8b86bf

where $X$ represents the set of points projected on the normal vector representing the slope of the hyperplane

- `normal_median`: This distribution is essentially the same as `normal_mean` but we use `median(X)` instead of `mean(X)`:

$$
	N(median(X),\eta \ std(X)
$$

^8b9c7b

- `scaled_uniform`: This is a [[EXPERIMENT SCRIPTS#^dbaab1|scaled version of the uniform distribution that is used in the EIF model]]:

$$
	U(\frac{min(X)}{\eta},\eta \ max(X))
$$
The `distribution` parameter can be set directly when creating a new instance of classes `ExtendedIsolationForest` or `Extended_DIFFI_parallel` or using the `set_distribution` method. This second solution is used so that it is easier to set this parameter getting its value as a command line argument. 

### `eta`

The `eta` parameter is simply the factor used in the definition of the distributions described [[EXPERIMENT SCRIPTS#`distribution`|above]]. Its default value, the one we used up to now in EIF+, is 2. 

Similarly to `distribution`, `eta` can be set directly when creating a new instance of `ExtendedIsolationForest` or `Extended_DIFFI_parallel` or using the `set_eta` method in case we are getting the `eta` value from the command line. 
