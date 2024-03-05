> [!note] Hyper parameters
>  - `models = [IF,EIF,EIF+,DIF,AutoEncoder]`
>  - `interpretation = [EXIFFI+,EXIFFI,DIFFI,RandomForest]`
>
# `contamination` experiments 

In the Contamination Experiments we do not need to divide in `scenario=1` and `scenario=2`. We do not want to evaluate the interpretation models, just the Average Precision values of the AD models as we increase the level of contamination in the training set. The contamination values we are are: `np.linspace(0.0,0.1,10)`. 

In these experiments we fix the dataset and for each one of them we produce the `precision vs contamination` plot for all the AD model we are comparing. 

For the models who also have an interpretation algorithm associated (i.e. `IF,EIF,EIF+`) we also perform the contamination experiment for the Global Importances. In this case we see how the Global Importances change as we change the level of contamination used to divide between inliers and outliers to compute the Global Importances. Essentially we compute the importances using the function `compute_global_importances` using different values for the input parameter `p`. 

- `wine` → ==ok==
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok 
	- `AutoEncoder` → ok

> [!summary] Comments on the results 
>  `EIF+` and `DIF` are the best with a clear decreasing behavior that saturates only in the last 3 points. `DIF` is a sligthly better because it starts from around 0.8 while `EIF+` starts at around 0.6.  `IF` and `EIF` start from 0.4 and there is an increase in Precision (for `EIF` they are more or less constant) passing from the first to the second point. The `AutoEncoder` is the worst one here since it starts from around 0.3. 

> [!note] Execution Times
> - Fit (s):
> 	 - `EIF+` → 0.06 
> 	 - `EIF` → 0.06
> 	 - `IF`→ 0.05
> 	 - `DIF` → $\approx$ 10 (with `n_estimators=200` → probably too much)
> 	 - `AutoEncoder` → 3.1
> 

- `glass` → ==ok==
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok 
	- `AutoEncoder` → ok

> [!summary] Comments on the results 
>  `EIF+` and `DIF` are the only ones showing a decreasing behavior with `DIF` that is a little better because it starts (at `contamination=0`) at 0.3/0.4 while `EIF+` starts at 0.2. The other models (`IF,EIF,AutoEncoder`) have more or less constant low Average Precision values (between 0 and 0.2). 
 
- `cardio` → ==ok==
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok 
	- `AutoEncoder` → ok

> [!summary] Comments on the results 
>  Also in `cardio` the best models are `EIF+` and `DIF`. `DIF` starts from something more than 0.8 and `EIF+` from about 0.7. The contamination plots show a continuously decreasing trend (there is not a proper saturation). 
>  The interesting result is the one of `AutoEncoder` where there is an increasing trend up to `contamination=0.04/0.05` ($5^{th}$/$6^{th}$ point in the plot) and then it starts decreasing. 

- `pima`
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok 
	- `AutoEncoder` → ok

> [!summary] Comments on the results 
>  Here all the Precision values are more or less the same independently on the contamination level. What changes between the different models is the Precision value reached. Unexpectedly the best ones are `EIF+,EIF` and `IF` with values aroun 0.6. On the other hand `DIF` has values between 0.4 and 0.6 and `AutoEncoder` 0.4. 

- `annthyroid`
	- `EIF+`  → ok
	- `EIF` → ok 
	- `IF` → ok → strangely this is better than `EIF,EIF+`
	- `DIF` → ok
	- `AutoEncoder` → ok

> [!summary] Comments on the results 
>  Here the Precision values saturate very fast and strangely the best model is `IF` since it starts from a Precision between 0.4 and 0.6. It is followed by `DIF` and `EIF+` (which start from 0.4) and then we have `EIF` and `AutoEncoder` starting from 0.2. 

> [!note] Execution Times
> - Fit (s):
> 	 - `EIF+` → 0.05
> 	 - `EIF` → 0.05
> 	 - `IF`→ 0.06
> 	 - `DIF` → $\approx$ 8 (with `n_estimators=6` → default value). This is a much lower value considering that `annthyroid` is a much bigger dataset with the respect to `wine`. 
> 	 - `AutoEncoder` → 31 → now that we have increase a lot the dataset size the execution time of `AutoEncoder` increased a lot. 
> 

- `breastw` → Re obtain the plots using `change_ylim` 
	- `EIF+` → ok 
	- `EIF` → ok 
	- `IF` → ok 
	- `DIF` → ok 
	- `AutoEncoder` → ok 

> [!summary] Comments on the results 
>  Very particular results. The Isolation based models (i.e. `EIF+,EIF,IF`) are very good. For all the contamination values the Precision is essentially 1 (in fact I have to modify the `plt.ylim`) parameter otherwise we cannot see the line of the plot. On the other hand `DIF` and `AutoEncoder` are worse. In `AutoEncoder` the situation is not so different in fact the values are more or less the same but around 0.9 Average Precision. On the other hand in `DIF` we are more or less around 0.6 for all the contamination values and with `AutoEncoder` 

> [!note] Execution Times
> - Fit (s):
> 	 - `EIF+` → 0.05
> 	 - `EIF` → 0.06
> 	 - `IF`→ 0.05
> 	 - `DIF` → $\approx$ 0.7 (it is still 10 times slower than `EIF+,EIF,IF`)
> 	 - `AutoEncoder` → $\approx$ 3.5 

- `ionosphere`
	- `EIF+` → ok 
	- `EIF` → ok 
	- `IF` → ok
	- `DIF` → ok
	- `AutoEncoder` → ok

> [!summary] Comments on the results
>  All very similar plots with more or less constant high Average Precision values. The ones at an higher point are `EIF+` and `DIF` (around 0.9/1), `EIF` and `IF` are at arond 0.8/0.9 and `AutoEncoder` is the worse around 0.7. 

> [!note] Execution Times
> Here the times are sligthly higher because `ionosphere` is the dataset with the highest number of features. 
> 
> - Fit (s):
> 	 - `EIF+` → 0.1
> 	 - `EIF` → 0.1
> 	 - `IF`→ 0.05
> 	 - `DIF` → $\approx$ 0.8 (it is still 10 times slower than `EIF+,EIF,IF`)
> 	 - `AutoEncoder` → $\approx$ 3.8 

- `pendigits`
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok
	- `AutoEncoder` → ok

> [!summary] Comments on the results
> Here the Average Precision values decrease in the first 3/4 point and then stabilize. The best models are, as usual, `EIF+` (starts from 0.4) and `DIF` (starts from about 0.5). Also `IF` is good  (starts a little before 0.4). `EIF` and `AutoEncoder` are a bit worse. In particular `AutoEncoder` has constant Precision all around 0.1 

> [!note] Execution Times
> 
> - Fit (s):
> 	 - `EIF+` → 0.07
> 	 - `EIF` → 0.07
> 	 - `IF`→ 0.06
> 	 - `DIF` → $\approx$ 7.8 (much higher here because `pendigits` is a big dataset)
> 	 - `AutoEncoder` → $\approx$ 30 

- `diabetes`

- `shuttle`

- `moodify`
	- `EIF+` → ok
	- `EIF`  → ok 
	- `IF` → ok
	- `DIF` → ok
	- `AutoEncoder` → ok

> [!note] Execution Times
> 
> - Fit (s):
> 	 - `EIF+` → 0.06
> 	 - `EIF` → 0.06
> 	 - `IF`→ 0.06
> 	 - `DIF` → $\approx$ 7.8 (similar to `pendigits`)
> 	 - `AutoEncoder` → $\approx$ 16 (for `moodify` I reduced the number of epochs to 50 → the time is more or less half than before). 

> [!summary] Comments on the results 
>  In `moodify` there is not a saturation in the Average Precision values but they continuously decrease as the contamination increases. The best model is `DIF` (that starts from about 0.8 of Precision), followed by `IF` (about 0.7) and `EIF+` (about 0.6). Finally we have `EIF` (about 0.6) and `AutoEncoder` (about 0.5). 
# General Comments on the results 

There is more or less a common trend across the different datasets → the only thing that changes are the Precision values → some datasets have higher values, some others lower. 

In any case the common trends are the following:

- As the contamination increases the Average Precision decreases → after some steps it saturates to a more or less constant value. This makes sense because ideally an AD model should be trained only on the inliers (so that it is able to learn the normal distribution) and then tests on both inliers and outliers so that it is able to distinguish the anomalies from the normal points because they "look different" from what it has seen in training. So showing some outliers in the training set the model is not looking at the exact inlier distribution since that is a little bit ruined from the presence of the outliers, the higher the number of outliers the worse it is.  
- In general the execution time is very high for the `DIF` model, moderately high for `AutoEncoder` and much faster for `EIF,EIF+,IF`. 
- In general `DIF` is the better model followed by `EIF+` and the others. This makes sense from a theoretical point of view. `DIF` is in fact able to isolate anomalies distributed also in non linear shapes. However from our results probably we can say that `EIF+` is the best one if we also include the time efficiency in the evaluation. In fact `EIF+` is much faster than `DIF`. Moreover in the paper we are presenting a way of interpreting the results of `EIF+` while, up to now, there are no interpretability methods for the `DIF` model. 
- Saturation in the Precision values in the plots depends on the contamination factor of the dataset (i.e. the percentage of outliers in the dataset). There are several datasets with a contamination factor lower than 0.1. So after their contamination level is passed the function `dataset.split_dataset(contamination=c)` will produce the same `X_train` independently on the value `c`. So the Precision values for all the values of `c` higher than `p` (contamination level  of the dataset) will be more or less the same (there will be some differences due to the randomness of the model). 
- The `AutoEncoder` model is most of the time the worst model by a significant margin. It may be because of the hyper parameters used? (in particular the `hidden_neurons` one)? Moreover, as we changed the parameters for `DIF` (because using `n_estimators` equal to the number of trees used in the forests does not make sense since in `DIF` `n_estimators` indicates the number of forests used) and we performed experiments on much bigger datasets, `AutoEncoder` is by far the slowest model. In fact it is a Neural Network like `DIF` but, differently from `DIF`, the layers here are trained (for 100 epochs). So maybe we can try to reduce the number of `hidden_neurons` or reduce the number of epochs to make it a little bit faster. In fact I do not think that the performances will change a lot if we decrease the epochs from 100 to 50. 