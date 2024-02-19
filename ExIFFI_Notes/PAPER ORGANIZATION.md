Let's draw here a possible structure of the new version of the paper with a detailed list. The idea is also to include the kind of plots we want to insert in each different section so that we have a clear idea of the experiments to carry on and have a clear idea on where exactly to put each different thing. 

We will start from the structure of the first version of the paper and re adapt it depending on the new things to add according to the suggestions added in the review. 

# Paper Outline

0.  **Abstract**
1. **Introduction**: Introduce the problem of Anomaly Detection and the need for interpretability and at the end report the structure of the paper saying what will appear in each different section 

> [!missing] 
> In the first version of the paper Section 2 is used to present the Isolation Based Approaches for Anomaly Detection: IF, EIF and DIFFI but we did not insert a Related Work section presenting other anomaly detection and interpretability methods for unsupervised learning. See [[ExIFFI PAPER REVIEW#Papers for Related Work|here]] for some papers that we may cite in the Related Work section.  

2. **Related Work** → Insert here citations to AD methods (e.g. DIF, PID, INNE, AutoEncoder) and interpretability methods used in Anomaly Detection. These methods are mainly used on DL based models (e.g. AutoEncoders) so here we may justify the importance of the introduction of an interpretability method on a very efficient for Anomaly Detection as IF/EIF/EIF+ are. Successively we can talk about IF/EIF and DIFFI but maybe removing all the detailed formulas used to describe them (to save some space since we have added the citations to other papers in the literature). 
		2.1 **Introduction of EIF+**: Introduce EIF+ with a more detailed description on how it is better than the other approaches 
3. **Interpretability for Isolation-based Anomaly Detection Approaches**: Description of DIFFI. Here we can also cite the paper by Mattia Carletti on the application of DIFFI on an industrial setting (***Interpretable Anomaly Detection for Knowledge Discovery in Semiconductor Manufacturing***) to justify the goodness of this model and why it is important ExIFFI (that is essentially a generalization of DIFFI). 
	3.2 **ExIFFI** → Explanation of ExIFFI with Global and Local Feature Importance
		3.2.1 **Visualizing Explanations** → Explanation of Bar Plot, Score Plot and Importance Map

> [!missing] 
> In this section we should cite the paper *Towards A Rigorous Science of Interpretable Machine Learning*  to justify the fact that we are not able to produce User Studies. 

4. [[PAPER ORGANIZATION#Experimental Results|Experimental Results]]
		4.1 Datasets 
			4.1.1 Synthetic Dataset → maybe add an explanation on how the synthetic datasets are used to evaluate the scalability of the proposed approach
			4.1.2 Real World Dataset → Here we may try to find new datasets with some semantic meaning in the features or we may try to study better the ones we already have
		4.2 [[PAPER ORGANIZATION#Ablation Study EIF+|Ablation Study EIF+]]
			- [[PAPER ORGANIZATION#Ablation Study EIF+ Plots|Ablation Study EIF+ Plots]]
			- [[PAPER ORGANIZATION#Scalability Experiments|Scalability Experiments]]
		4.3 [[PAPER ORGANIZATION#Performance Report|Performance Report]]
			- [[PAPER ORGANIZATION#Performance Report Plots|Performance Report Plots]]
		4.4 Experimental Evaluation of ExIFFI 
			4.4.1 Performance Evaluation Strategy → Proxy Task Feature Selection 
			4.4.2 [[PAPER ORGANIZATION#Ablation Study of ExIFFI|Ablation Study of ExIFFI ]]
				- [[PAPER ORGANIZATION#Ablation Study ExIFFI Plots|Ablation Study ExIFFI Plots]]
				- [[PAPER ORGANIZATION#Scalability Experiments|Scalability Experiments]]
			4.4.3 Experiments Synthetic Datasets → Same as in first version but with best ExIFFI model obtained from Ablation Studies 
			4.4.4 Experiments Real World Datasets → Same as in first version but with best ExIFFI model obtained from Ablation Studies
5. **Conclusions** 
				
## Experimental Results 
### Ablation Study EIF+ 

> [!todo] Before or after the Comparison with the other AD models?
> Decide weather to put the experiments regarding the Ablation Study before or after the Comparison with the other AD models. I think it is better to put it before the Experimental Results so that  we can use the best configuration of parameters for EIF+ in the Comparison. 
> 

Values to try for the parameters: 

- ~~Number of trees `n_trees = [100,300,600,800]` → default: `300`~~ 
- ~~Data Normalization Scalers `scaler = [StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler]` → default: `StandardScaler`~~
- Distribution → default: `normal_mean` (or `normal_median` if it is better)
	- [[EXPERIMENT SCRIPTS#^8b86bf|`normal_mean`]]
	- [[EXPERIMENT SCRIPTS#^8b9c7b|`normal_median`]]
	- [[EXPERIMENT SCRIPTS#^418327|`scaled_uniform`]]
	- [[EXPERIMENT SCRIPTS#`normal_range`|`normal_range`]] 
	- [[EXPERIMENT SCRIPTS#`normal_MAD`|`normal_MAD`]]   
- Values of the scaling parameter $\eta$ → `eta = [2,2.5,3,3.5,4]` → default: `2`
- Contamination level used in `X_train` → `contamination = np.linspace(0.01,0.1,10)` → default: `0.1`

#### Ablation Study EIF+ Plots 

To evaluate the different parameters combinations I think it is better to focus on the Average Precision and ROC AUC Score metrics 

1. Plot of Average Precision/ROC AUC for different values of (and all the other parameters at the default value)
	-  `n_trees`
	- `scaler`
	- `distribution`
	- `eta`
	- `contamination`
	- Obviously we have also to include the case with all the default values → `[n_trees:300,scaler:StandardScaler,distribution:normal_mean,eta:2,contamination:0.1]`
### Performance Report  

In the previous version of the paper we used two scenarios to design the training and testing. Now we can add a third one:
- Scenario I → Train and test on the entire dataset
- Scenario II → Train on inliers and test on the entire dataset
- Scenario III → It is very similar to Scenario II but we insert progressively insert some outliers in the training set. We can try with different values of contamination factor. For example we can try with a range of 10 values going from 0.01 to 0.1 → `np.linspace(0.01,0.1,10)`

For each one of these three scenarios we have to compute the Performance Report of all the models under consideration:
- IF
- EIF
- EIF+
- DIF
- AutoEncoder
- ECOD ? 

The Performance Report is composed of the following metrics:
- Precision 
- Recall
- `F1` Score
- Accuracy
- Balanced Accuracy
- Average Precision 
- ROC AUC Score 
#### Performance Report Plots 

1. Vertical Bar Plot: 
	- Average Precision/ROC AUC vs Scenario I, II, III (here we may use the best value among the different ones produced by Scenario III)
	- Average Precision/ROC AUC vs different values produced by Scenario III 
	- Average Precision/ROC AUC vs different model (on the same Scenario)
2. Violin Plot with the Average Precision?
3. Table with all the numeric results → in Appendix 

In the first version of the paper we used a table to compare the Average Precision results (Table 2) but here with all these metrics and all these different models it may be a mess. We can still create a table but probably it is better to insert it into the Appendix. 

So we may use some plots to represent how the values of a certain metric (e.g. ROC AUC and Average Precision are the most important ones) change w.r.t the different models and w.r.t. the different scenarios. 

### Ablation Study of ExIFFI 

Values to try for the parameters: 

- Number of trees → `n_trees = [100,300,600,800]` default: `300`
- Contamination factor used to divide between inliers and outliers in `global_importance` → `contamination = np.linspace(0.01,0.1,10)`
- Depth Based Score or not: `depth_based = [True,False]` default: `False`
- Different number of runs for the GFI Computation in `compute_global_importance` method → `n_runs = [5,10,15]` default: `10`

> [!note] 
> Use $AUC_{FS}$ as the quantitative metric to evaluate the goodness of the interpratations. 
> Probably we need to work only on synthetic datasets where there is a clear ground truth on what are the most important features. 
> Metric → [Non Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
#### Ablation Study ExIFFI Plots 

To evaluate the goodness of different parameters configurations of ExIFFI we use a modified version of the Feature Selection Proxy Task: 

- Produce the Feature Selection Plot using the same strategy used in the first version of the paper (i.e. every time remove the least important feature)
- Produce the Feature Selection Plot using the opposite strategy (i.e. every time remove the most important feature)

At the end we compute the value of the area in between the two curves →the higher it is the better is the interpretation returned by the model. 

So we can produce plots representing how this metric, that can be called $AUC_{FS}$ (where $FS$ stays for Feature Selection), behaves changing: 

- `n_trees`
- `contamination`
- `depth_based`
- `n_runs`
- All default parameter values: `[n_trees:300,contamination:0.1,depth_based:False,n_runs:10]`

### Scalability Experiments 

These experiments should be done for both EIF+ and ExIFFI 

Use synthetic datasets, as done in the DIF paper, to see how the model scales to dataset of increasing size and increasing dimensionality. Use the usual synthetic dataset and perform experiments computing the execution time: 
	- Use the synthetic datasets `Xaxis,Yaxis,Bisect,Bisect3D,Bisect6D` and change the number of samples contained in them keeping constant the dimensionality (`n_features = 6`) → `n_samples = np.linspace(1000,256000,10)`)
	- Use synthetic datasets `Xaxis,Yaxis,Bisect,Bisect3D,Bisect6D` and change the number of features keeping constant the number of samples ( `n_samples = 1000`) (e.g. `np.linspace(1000,256000,10)`). Here since we are adding multiple features we can also increase the number of dimensions along which to distribute anomalies (e.g. `Bisect10D,Bisect20D,Bisect30D,...`)

> [!note] 
> Probably we will have to say that we are taking inspiration from the DIF paper for this scalability experiments. 
> 

> [!note] 
> The goal of these experiments is just to evaluate the execution time of the algorithm, not to evaluate its performances on these datasets.

> [!important] Plots
>  The plots will be plots of:
>  - `n_samples` vs `execution_time`
>  - `n_features` vs `execution_time`

### ExIFFI vs TreeSHAP 

- Find an implementation of TreeSHAP for Anomaly Detection 
- Compare the execution times of ExIFFI and TreeSHAP with different datasets with increasing complexity 

