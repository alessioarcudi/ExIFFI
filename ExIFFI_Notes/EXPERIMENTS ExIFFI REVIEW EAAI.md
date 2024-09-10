Let's use this note to keep track of the experiments I am running for the resubmission of `ExIFFI` after the review from `EAAI`. 
# Experiments to perform

 - [x] Time Scaling Experiments for `KernelSHAP`
 - [x] Experiments for the `ECOD` AD model
 - [x] Correlation between Feature Importance and Anomaly Score 
 - [x] New synthetic dataset `bisect_3d_prop` experiment
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
	- `Xaxis_100000_6` → ==ok==
- Datasets varying number of features and with 5000 samples
	- `Xaxis_5000_16` → ==ok==
	- `Xaxis_5000_32` → ==ok== 
	- `Xaxis_5000_64` → ==ok== 
	- `Xaxis_5000_128` → ==ok==
	- `Xaxis_5000_256` → ==ok==
	- `Xaxis_5000_512` → ==ok==
# Experiments `ECOD` model 

Use the `PyOD` implementation of [ECOD](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.ecod). The `ECOD` model has some sort of interpretation but it is only local and only a graphical interpretation if I do remember correctly thus there should not be something like the `LFI` or `GFI` score to exploit for a comparison with `ExIFFI`. So we can use it as we used `DIF` and `Autoencoder`. So we will use it in the experiments:

- [x] Precision metric experiment
- [x] Contamination experiment
- [x] Time Scaling `fit-predict` experiment 
- [x] `GFI` experiments with the new `ECOD` Feature Importance 
- [x] Local Scoremaps experiments 
- [x] Feature Selection Experiments 
	- [x] Fix the `Xaxis` plot with `change_ylim`
- [x] Time Scaling Experiments for the `importances` Time Scaling plot 

## `ECOD` Feature Importance 

According to the `ECOD` paper, and also to the `AcME-AD` paper, this model is intrinsically interpretable and so we can use it as another method to compare to our `EXIFFI,EXIFFI+` in terms of interpretability performances. However the concept of interpretability presented in the `ECOD` paper is not very similar to the one of `DIFFI,EXIFFI,EXIFFI+`. 

In fact in the paper they present the so called *Dimensional Outlier Graph* which is dedicated to a single sample and simply reports the outlier scores for each feature (i.e. the value of the `ECDF` function for each feature) compared with the 99% percentile of that feature. 

In order to obtain something similar to the `GFI` score we use to obtain the Score Plot, Bar Plot and to perform the Feature Selection experiments we worked a little bit with these outlier scores for each feature, which we can call $0_i^{(j)}$ (i.e. outlier score of feature $j$ for sample $i$). 

Given an object representing the `ECOD` model in Python we can access these outlier scores with the attribute `O`. 

> [!note] 
> For some reason the importance matrix is doubled. So if I have a dataset $\mathcal{X}$ with shape `(100,10)` the `np.array` I obtain in output doing `model.O` will have shape `(200,10)` and the element in index 0 will be the same as the one with index 100.  

Commenting the *Dimensional Outlier Graph* in the paper they say that the most important features (i.e. the most anomalous features) are the ones closer to the 99% percentile (so features on which that specific sample has very high values) → so we want to give an high importance score to these features and low importance scores to features which are far from this percentile. 

 Let's consider to have $p$ input features and $n$ input samples, thus the input dataset is $\mathcal{X} \in \mathbb{R}^{n \times p}$. 

So a possible way of assigning importance scores is to use the inverse of the distance between the  outlier scores and the 99% percentile of the feature outlier score we are considering (i.e. $0_i^{(j)}$). Thus the importance of feature $j$ for sample $i$ will be:

$$
	I_i^{(j)} = \frac{1}{1+(\alpha_{0.99}^{(j)} - 0_i^{(j)})^2}
$$
where $\alpha_{0.99}$ is the 99% percentile of the outlier scores for feature $j$. We use the squared distance to avoid having negative values. We also add 1 to the squared distance to avoid having the importance exploding to $\infty$. In fact it may happen that the value of feature $j$ for sample $i$ coincides with $\alpha_{0.99}^{(j)}$ and thus the distance is 0. In this specific case $I_i^{(j)}$ will be 1 that is the maximum `LFI` score achievable. 

This is the Local Feature Importance score of feature $j$ for sample $i$. Putting together the scores for all the features $j=1,\dots,p$ we obtain the `LFI` score for sample $i$ (with $i \in \{1,\dots,n\})$: 

$$
	I_i = [I_i^{(1)},I_i^{(2)},\dots,I_i^{(p)}] \in \mathbb{R}^p
$$

Now to obtain the `GFI` score we can do the same thing that is done in `DIFFI` and `EXIFFI`. We divide the dataset into inliers and outliers according to the predictions done by the `AD` mode (i.e. `ECOD` in this case) and we compute the `LFI` scores separately on set of predicted inliers $\mathcal{P}_I$ and on the set of predicted outliers $\mathcal{P}_O$. 

> [!note] 
> In this step when we do the predictions (to distinguish between inliers and outliers) we have to pass a contamination factor (i.e the percentage of anomalies in the dataset). This is used in the definition of the threshold applied on the Anomaly Score to perform the predictions. Here we can be in two scenarios:
> - Unsupervised setting → In this case we have no labels so we have no idea on how much anomalies are there in the dataset so we usually try to guess a feasible value for the contamination, usually we use 0.1 → **this is the contamination I used in all the `GFI` experiments done for the paper**.  
> - Supervised setting → In the case ground truth information are available we can easily compute the percentage of anomalies in the dataset and thus it makes sense to use this value as the contamination. In this way the model will always predict a number of anomalies that is equal to the true value (but it can in any case make errors). In the metrics experiment in the paper we can access the labels and thus in that case I used `p=dataset.perc_outliers`. 

Then we put together the inliers and outliers importances into two $p$ dimensional vectors with the mean:

$$
	\hat{I}_I = \frac{1}{N_I} \sum_{x \in \mathcal{P}_I} I_x
$$
$$
	\hat{I}_O = \frac{1}{N_O} \sum_{x \in \mathcal{P}_O} I_x
$$

> [!note] 
> In this case, differently from `DIFFI` and `EXIFFI`, we do not have to normalize divide by a counter or by the normal vectors because there is no randomness in the algorithm and thus there is no risk of being biased towards a certain feature → also because the features are considered independent and the $O_i^{(j)}$ computation are done independently

Finally the `GFI` score is computed as:

$$
	GFI = \frac{\hat{I}_O}{\hat{I}_I}
$$

> [!important] 
> Since the model is deterministic in determining the importance scores it does not make sense to perform multiple runs in the `GFI` experiments → they will be all the same.  

## `ECOD` Performances

Looking at the results obtained with the Contamination plots we can see how `ECOD` has similar performances to the other model only on the `bisect` datasets while on the others the performances are not very good. A possible explanation for this behavior may be connected to the fact that `ECOD` makes the assumption that the **features in the dataset are independent**. In fact it computes the Anomaly Scores individually on each feature (using the `ECDF` function estimates on each single feature) and then combines all the univariate scores to obtain a single score for each sample. The problem is that this assumption likely does not hold on the benchmark datasets we used in this paper. This is certified by the fact that in most cases there are multiple features with high importance values. On the other hand in the synthetic datasets we can safely say that the features are independent (because we generated each one of them sampling from a certain distribution) and in fact in the `bisect` datasets the model works decently. The strange thing are probably the bad performances on `Xaxis` and `Yaxis`. 

## `ECOD` Time Performances 

Looking at the new Time Scaling plots for the `fit` and `predict` operations (that now include also the `ECOD` model) we can observe the following things:

 - `samples` → In general `ECOD` is the fastest model considered but as the number of samples increases its computational time increase up to surpassing or becoming very close to the times of the isolation based models (i.e. `IF,EIF,EIF+`)
 - `features` → Probably because of the exploitation of parallel computing in the Anomaly Score computation the computational time stays more or less constant as the number of features increases. Only in `predict` it is possible to notice a clear increase in time (from 256 to 512 features) probably due to the fact that after having computed the univariate Anomaly Score on all the features these values have to be combined in some way and that operation scales linearly with the data dimensionality. 

> [!note] 
> Moreover in the Introduction to the [ECOD paper](https://arxiv.org/abs/2201.00382) the authors says that as the number of features increases there are some technical difficulties in using `ECDF`s to estimate the data distribution. In fact when we increase the dimensionality the `ECDF` converges slowly to the true `CDF` of the data. They solve this problem using the approach of dividing the joint `ECDF` into $p$ univariate `ECDF`s, where $p$ is the number of features. 
> In this step they assume that the features are independent. In fact to combine the tail probabilities from the different features they simply multiply them (like if we have 2 independent random variables → we can compute the intersection probability as the product of the single probabilities). 

## Comments on `GFI` experiment results 

Some very fast comments looking at the Score Plots returned by the `GFI` experiments performed on the `ECOD` interpretation model. 

> [!note] 
> If I just write ==ok== it means that the result make sense and are good compared to expectation 

### Synthetic Datasets 

| Dataset          | Description                                                                            |
| ---------------- | -------------------------------------------------------------------------------------- |
| `Xaxis`          | No sense, because `ECOD` is very bad on `Xaxis` in the metrics experiments             |
| `bisect`         | For some reason it selects Feature 0,1,2, as the most important (and not only 0 and 1) |
| `bisect_3d`      | ==ok==                                                                                 |
| `bisect_3d_prop` | ==ok==                                                                                 |
| `bisect_6d`      | ==ok==                                                                                 |
### Real World Datasets

> [!note] 
> Remember that in the Real World datasets we have to use `pre_process` 

| Dataset      | Description                                                                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wine`       | Not aligned with the Score Plot of `EXIFFI+` and `DIFFI` but `ECOD` was very bad on `win                                                                                                                |
| `breastw`    | Makes sense, Feature 8 most important followed by 4 and 6. In any case `ECOD` was good on `breas                                                                                                        |
| `annthyroid` | Feature 3 is on top (this is the top 2 in `EXIFFI+`) but Feature 1, which is very important in `EXIFFI+`, is among the last ones. The performances of `ECOD` on `annthyroid` were not very good though. |
| `pima`       | Similar to `EXIFFI+`: it has `Insule,BMI` on top while `EXIFFI+` had `Blodd_Pressure,Insuline                                                                                                           |
| `cardio`     | In `scenario_2` there is Feature 6 on top like in `EXIFFI+`, then top 2 and 3 are 17                                                                                                                    |
| `glass`      | The most important features (i.e. `Ba,K`) are not on the top spots. In any case `ECOD` was good on                                                                                                      |
| `ionosphere` | The most important feature is 0, differently from `EXIFFI+` where it is 1 but the result is the same as                                                                                                 |
| `pendigits`  | Very similar to `EXIFFI+` with Feature 3 and                                                                                                                                                            |
| `diabetes`   | Here we have `bmi` on top while on `EXIFFI+` we have `HbA1c_level` and `blodd_gluc                                                                                                                      |
| `shuttle`    | Not very similar, there is only Feature 0 that is in the top 3 like i                                                                                                                                   |
| `moodify`    | The most important features are `energy` and `loudness` while in `EXIFFI+` it'                                                                                                                          |
# Experiments Correlation

This experiment was proposed by [[ExIFFI PAPER REVIEW EAAI#Reviewer 2|reviewer 2 comment 6]] and it may be considered as another experiment to evaluate the effectiveness of the `ExIFFI` interpretation algorithm. The idea is that since the importance scores we obtain with `ExIFFI` should quantify the relevance of certain features in detecting anomalies then **samples with high feature importance values should also have an high Anomaly Score**. We can quantify this effect computing the correlation between the importance scores and Anomaly Scores for each sample. We can use the following approach: 

- For each different interpretation algorithm under comparison (i.e. `ExIFFI,ExIFFI+,DIFFI,RandomForest`)
	- For each dataset
		- Compute the `LFI` of all the samples 
		- Compute the sum of the `LFI` scores over all the features
		- Compute the Anomaly Score of the `AD` method (use the `predict` method that returns the Anomaly Scores for all points)
		- Compute the correlation 

At then end, add a column with the correlation values and add it to the $AUC_{FS}$ table.  

- [x] Correlation experiments on `EXIFFI+,EXIFFI,DIFFI,IF_EXIFFI` on `scenario_2`
- [x] Correlation experiments on `EXIFFI+,EXIFFI,DIFFI,IF_EXIFFI` on `scenario_1`

## Datasets missing the correlation experiments 

- [x] `scenario_2`
	- [x] `ionosphere`
	- [x] `glass`
	- [x] `cardio`
	- [x] `bisect_3d_prop`
	- [x] `bisect`
- [x] `scenario_1`
	- [x] `bisect` (only `IF_DIFFI`)
	- [x] `bisect_3d`
	- [x] `bisect_3d_prop`
	- [x] `bisect_6d`
	- [x] `annthyroid`
	- [x] `breastw`
	- [x] `cardio`
	- [x] `diabetes`
	- [x] `glass`
	- [x] `ionosphere`
	- [x] `moodify`
	- [x] `pendigits`
	- [x] `pima`
	- [x] `shuttle`
	- [x] `wine` 
^c65818
# New synthetic dataset `bisect_3d_prop` experiment

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

- [x] `GFI` experiment for (only the ones needed for the paper) → redo for `bisect_3d_skewed` (`EIF+_EXIFFI+` already done)
- [x] Local Scoremaps `EXIFFI+` (only the ones needed for the paper) → redo with `bisect_3d_skewed`
- [x] `ablation_EIF+` experiment 
- [x] Contamination Experiment 
- [x] Feature Selection Experiment (only the ones needed for the paper) → redo with `bisect_3d_skewed`
- [x] Metrics Experiment → redo with `bisect_3d_skewed`
- [x] Time Scaling Experiment  
	- [x] → use the same as `bisect_3d_prop_old` (the dimension of the dataset is the same only the value change )

# Final things TO DO 

- [x] Insert captions on the paper where it is written <span style="color:red">CAPTION DA FARE</span>. In particular the captions need to be done for the following images:
	- [x] Full Contamination plot of `Xaxis` and `bisect_3d_skewed`
	- [x] Score Plot + Local Scoremap of `Xaxis` and `bisect_3d_skewed`
	- [x] Full Contamination plot of `glass`
	- [x] Score Plot + Local Scoremap of `glass`
- [x] Update the bold numbers in the:
	- [x] Average Precision Table (put in bold the best Average Precision values for each different dataset)
	- [x] $AUC_{FS}$ score table 

# New paper titles

[[ExIFFI PAPER REVIEW EAAI#Reviewer 3|Reviewer 3]] pointed out that the title of the paper is a bit confusing. That's one of the drawbacks of introducing two novel contributions (i.e. `ExIFFI` and `EIF+`) in a single paper. It would have probably been simpler to do two different smaller papers, one for `ExIFFI` and one for `EIF+` but at this point we have to stick with what we have. 

Possible new title ideas given by ChatGPT:

- **"ExIFFI: Interpretable Anomaly Detection via Enhanced Isolation Forest Feature Importance"**
- **"Explaining Anomalies: A Novel Approach with ExIFFI and EIF+"**
- **"Improving Anomaly Detection Interpretability with ExIFFI and EIF+"**
- **"ExIFFI and EIF+: A Comprehensive Framework for Explainable Unsupervised Anomaly Detection"** → **N.B.**
- **"Enhancing Isolation Forest for Generalized Anomaly Detection and Interpretability"** → **N.B.**
- **"Unveiling Anomalies: Feature Importance and Generalization in Isolation Forest Models"**
- **"ExIFFI: Bridging the Gap Between Anomaly Detection and Interpretability in Isolation Forests"**
- **"EIF+ and ExIFFI: A Unified Approach for Generalized and Interpretable Anomaly Detection"** → **N.B.**
- **"A Novel Interpretability Framework for Isolation Forest Anomaly Detection: ExIFFI and EIF+"**
- **"Feature Importance in Unsupervised Anomaly Detection: The ExIFFI Approach to EIF+"**

# Cover Letter 

In the Cover Letter we can keep track at all the comments left by the reviewers and se which one of them we have solved and which not to see what are the little things missing. Here I will write the things that we are still missing:

- Comment 4 reviewer 1 → Strange comment on the old citations 16,17,18 → it is not clear why he/she thinks that citations 16 is not in the same format of citations 17 and 18
- Comment 14 reviewer 1 → Here since Feature 0 is the most important one in the synthetic datasets and in some of the real world one he thinks that it is always the most important. In any case discussing about the impact of feature importance in real-world scenarios should solve the comment
- Comment 3 reviewer 2 → Probably here we should create a dedicated section to go deeper in explaining why `EIF+` is better than `EIF` putting together the results coming from the other sections. `EIF+` is though to work well in Scenario II, so, taking in consideration the paper talking about the *normality assumption*, when the training data are composed of only inliers. Results are on our side because from the Average Precision table if we compute the mean Average Precision across all the datasets we can see how the best AD model in Scenario I is `EIF` while `EIF+` is the best one in Scenario II. 
- Comment 6 reviewer 2 → Chiara suggests to also insert the `NDCG` metric as a quantitative evaluation of the interpretation algorithms. This can be done for the synthetic dataset where we have a ground truth on the most important features → the main problem is to how exactly decide the what is the correct ground truth feature ranking. For example in `Xaxis` we only care that the model is able to put Feature 0 at the first place, since the remaining 5 features are just noise we do not care about their order. 
- Comment 1 reviewer 3 → Here it suggests to change the title and the content to make the paper simpler to follow. With the modifications we did making the paper shorter it may probable be simpler to read. For what concerns the titles there are some ideas from us and also from [[EXPERIMENTS ExIFFI REVIEW EAAI#New paper titles|ChatGPT]] we can consider.
- Comment 7 reviewer 3 → Search the papers from the `C-RISE` research group the reviewer is referring to → in fact he/she just mentions the research group and the topics of the papers (i.e. Bayesian Network-based methods, scores of fault diagnosis and isolation models, methods extended to complex engineering, system safety and risk analysis) without giving us the titles and authors
- Comment 1 reviewer 4 → Verify that the numerical results reported in the Abstract coincide with the results showed in the big tables → OK 
- Comment 9 reviewer 4 → Elaborate on the potential limitations of your work → Say that the model is model specific and, as a consequence, it works only on isolation based AD models → for `EXIFFI`. For `EIF+` → it's surely better than `IF` but it still uses linear cuts and thus it may struggle in detecting anomalies with complex non linear shapes. 
- Comment 10 reviewer 4 → *More carefully proofread your manuscript* → Do not really understand what they mean here → re read the paper multiple times to avoid any orthographic error.

## C-RISE Papers to cite

The C-RISE research group focuses on Risk Assesment, Risk Management, Security, Fault Detection and all these kind of stuff. They focus on offshore oil and gas as well as mining industries operating in harsh and remote environments including the Arctic and Canadian North. This topic can be considered connected to Anomaly Detection because AD and Fault Detection are similar and in any case anomalies can signal potential risks to avoid and manage. 

The head of the research group is Faisal Khan, so searching him on Google Scholar I found these papers:

- [Safety analysis in process facilities: Comparison of fault tree and Bayesian network approaches](https://www.sciencedirect.com/science/article/pii/S0951832011000408?casa_token=b6iwYsUpnZsAAAAA:plNa_yjewrrIHiOri7Yce36tbiuoP-6HxYy7Cmed4wfjQqYJ4e28MnezKiRfVNKKENIFQvaEmw) (August 2011) → Here they compare Bayesian Networks to Fault Trees as two safety analysis methods. Fault Trees is a method developed for Accident Analysis that determines the potential causes of an undesired event (i.e. the top event) using a tree that represents the logical connections between primary events that lead to the top event. This is somehow similar to what our Feature Importance score do → giving a feature ranking we provide a list of the most probable causes under an anomalous point. In the paper they highlight some limitations of Fault Trees and introduce the usage of Bayesian Networks as an alternative method for this task. 
- [Methods and models in process safety and risk management: Past, present and future](https://www.sciencedirect.com/science/article/pii/S0957582015001275?casa_token=7t1TMp8XgyMAAAAA:6O3WWVO1k4aHhl8P7NzrqDGqcjZ6MfqY1atwV6cdkyNAmeD-H0kXk1zb1UWiLGaddbz9B9dj_A) (November 2015) → This is survey on risk analysis, process safety and risk management tools. 

Another member of the C-RISE research group is Syed Imtiaz. Among its papers on Google Scholar we find:

- [An analysis of process fault diagnosis methods from safety perspectives](https://www.sciencedirect.com/science/article/pii/S0098135420312400?casa_token=cyOW0iNQr6IAAAAA:AQMnd_jqx2KSi5Q_Eo5MOxc0goWR5Z6jJxTCdsBRuvbN5fpUZsojiMGD-qmiI3gGjF0I3w5hSw) (February 2021) → This is another survey that talks also about Industry 4.0. They focus on three essential aspects of process safety: Fault Detection and Diagnosis (FDD), Risk Assessment (RA) and Abnormal Situation Management (ASM). This survey is very interesting because it also talks about Anomaly Detection when it talks about Fault Detection and thus we can include it in the Introduction when we talk about the different applications of Anomaly Detection. 
- [A data-driven Bayesian network learning method for process fault diagnosis](https://www.sciencedirect.com/science/article/pii/S0957582021001804) (June 2021) → This is probably the paper the reviewer was referring to when he/she wrote *Bayesian Network-based models*. It introduces a new data driven Fault Detection and Diagnosis (FDD) model that combined PCA with Bayesian Networks. We can insert this citation in the Related Work section when we talk about the Bayesian Network approach. 
- [Fault detection and pathway analysis using a dynamic Bayesian network](https://www.sciencedirect.com/science/article/pii/S0009250918307371) (February 2019) → More or less a similar work to the previous one. Here they also talk about a Bayesian anomaly index (DBAI) based control chart. 

