
Final little modifications of `ExIFFI` before new submission to `EAAI` Journal. 

- [ ] Summarize the paper
	- [ ] Remove `DIFFI` explanation → not necessary
	- [ ] Rephrase the role of `ExIFFI` as an Interpretability method that work for different versions of Isolation Forest (i.e. `IF`, `EIF` and `EIF+`)
	- [ ] Move one (or more) datasets from the `Experimental Results` sections to the `Appendix` (e.g `Bisect6D`) to reduce the number of pages and making the `Experimental Results` section less redundant and repetitive
	- [ ] Summarize some explanations. For example with the new Overleaf template of `EAAI` new the Local Scoremaps description is one page long → too much. 
- [x] Additional Experiments (for the moment only on the datasets inserted in the paper)
	- [x] To prove the usability of `ExIFFI` as an Interpretability method for `IF` based models we need to try to see what happens if we use it to interpret the `IF` model. So we have to use the `IsolationForest` class (contained in `EIF_reboot.py`) as the AD model to explain. 
		- [x] Produce the Bar and Score Plot 
		- [x] Produce the Feature Selection Plot and compute the new `AUC_FS` values
		- [x] Produce the Local Scoremaps 
	- [x] Add the new produced Feature Selection Plot in the paper
	- [x] Try to compute the GFI scores again using as contamination the contamination of the dataset (i.e. `dataset.perc_outliers`) instead of the default value of 0.1. I hope this does not change much otherwise I should redo all the experiments for all the datasets → This thing makes sense but theoretically we cannot do it because we are in an unsupervised setting. We have the labels only because we are using some benchmark datasets built ad-hoc but in a real world scenario typically we do not have them. In the Industrial `ExIFFI` paper we used the dataset contamination because it was kind of a supervised dataset (since `TEP` is actually a synthetic dataset).

## Experiments

Keep track of the experiments on `IF_EXIFFI`. Let's starts first with the datasets included in the main part of the paper: 

- Real World Datasets
	- `glass`
	- `annthyroid`
	- `moodify`
- Synthetic Datasets
	- `Xaxis`
	- `bisect_3d`
### Real World Datasets

- [x] `glass`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

Importance plots very similar to the ones of `EXIFFI` and `DIFFI` but the Feature Selection Plot is way better. The red line is similar (it goes from 0.7 up to almost 1) but the blue one decreases much better and it does not have a peak like the one from 5 to 4 features there is in `EIF+_EXIFFI+`. In `scenario1` both are bad, as expected. 

- [x] `annthyroid`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

Importance plots more similar to `IF_DIFFI`. Differently from `EXIFFI+` where the bars are more multicolored from the second position onwards in the Bar Plot here the bars after Feature 1 are dominated by one color (e.g. last two bars clearly dominated by Feature 0 and 4). In `scenario1`, as it happens in `DIFFI`, `IF` is still able to clearly identify Feature 1 at the first place in the ranking. 

Feature Selection Plots almost identical to the ones of `EXIFFI+` and `DIFFI`. 

- [x] `moodify`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

`IF_EXIFFI` here it is clearly better than both `EXIFFI+` and `DIFFI` in the sense that is able to identify `loudness` as the most important variable in all the runs → `EXIFFI+` identifies `loudness` most of the times (but around 70% of the times) while `DIFFI` was really bad in `moodify`. In this case `IF_EXIFFI` is more similar to `IF_RandomForest` and `EIF+_RandomForest`. This similitude is confirmed looking at `scenario1` where `spec_rate` is the most important feature but not with a huge margin on the others. 

The Feature Selection Plot is much better, similar to the ones using `RandomForest` (that were the best ones with `moodify`). 

### Synthetic Datasets

- [x] `Xaxis`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`
 
Here the Importance Plots in `scenario_2` are very bad as it happens in `DIFFI` and this is expected because of the `IF` artifacts that are exposed in the `Xaxis` (and `Yaxis`) synthetic datasets. The only strange thing is that it is very bad also in `scenario1` where `DIFFI` was able to identify Feature 0 as the most important one. 

As a consequence the Feature Selection plots are the symmetric version of the one of `EXIFFI+` in the sense that the red area occupies almost the entire figure but the red and blue line are swapped and so the `AUC_FS` is negative. Essentially this means that the model completely missed the importance order, in fact in the Importance plots we can see that Feature 0 is ranked in the last position around 50% of the times. 

- [x] `Bisec3D`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

The Importance Plots are similar to the ones of `EXIFFI+` and `DIFFI`. With the new distribution associated with `bisect_3d` `IF` is not affected by its artifacts and so it is able to correctly identify Features 0,1,2 as the most important ones. 

As a consequence also the Feature Selection plots are very similar to the ones of `EXIFFI+`. 

## Experiments Appendix Datasets 

### Real World Datasets

- [x] `breastw`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

Similarly to `DIFFI` it is able to identify Feature 8 as the most important in all the runs, while `EXIFFI` is not good in doing that in this dataset. In fact `breastw` is one of the dataset with an high contamination factor (52%) (pretty abnormal for the Anomaly Detection task) and moreover inliers and outliers are distributed in a grid like shape, so it's easier to detected them using the axis-aligned cuts typical of `IF`. 

Feature Selection Plots very similar to the ones of `DIFFI` and `EXIFFI`. 

Mean Importance time → 0.10

- [x] `cardio`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

It is similar to `EIF_RandomForest` and `EIF+_RandomForest` that detect in all the runs Feature 6 as the most important. Differently from the previously cited models Feature 6 is detected also in `scenario1`. 

In the Feature Selection plot is more similar to `DIFFI` than to `EXIFFI` that is much better in this case. 

Mean Importance time → 0.36

- [x] `diabetes`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

More similar to `DIFFI` since they both identify `bmi` as the most important feature. `IF_EXIFFI` however identifies `HbA1c_level` and `blood_glucose_level` as the most important in `scenario1` as it is done by `EXIFFI`. 

In the Feature Selection plots it is similar to `DIFFI` in `scenario2` (because it misses the first two most important features) with a negative `AUC_FS` value but it is better in `scenario1` because there it identifies the two most important features (`AUC_FS` positive). 

Mean Importance time → 0.73

- [x] `ionosphere`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

Very differently from all the other models `IF_EXIFFI` identifies Feature 0 as the most important in all the runs. Also the `RandomForest` based models identify a single dominant feature but they are different: Feature 25 for `EIF+_RandomForest` and `EIF_RandomForest` and Feature 31 for `IF_RandomForest`. Here the high-dimensionality of the dataset and the contamination factor (35% with the respect to the 10% used by default) makes the interpretability task more challenging. 

Feature Selection plots in any case very similar. Here we have the typical situation that can be observed when there is an high-dimensional dataset (like it happened with `TEP`). At the beginning the Average Precision is almost constant and then it starts to decrease in the last features because there is a subset of important features and as we start to remove some of them from the group the model performance start to drop. Interestingly in `scenario1` the `AUC_FS` metric is positive and the Average Precision increases a lot passing from 2 to 1 features, so probably Feature 0 is correctly identified as an important feature. 

Mean Importance time → 0.16

- [x] `pendigits`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

Pretty randomic distribution of the importances but in `IF_EXIFFI` Feature 3 and 5 are clearly the first two. These two features are in the top 4 both for `DIFFI` and `EXIFFI` so that's a good sign. 

Feature Selection Plots more or less similar to the ones of `EXIFFI`. 

Mean Importance time → 1.13 

- [x] `pima`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

Situation similar to the one of `ionosphere`, in fact `IF_EXIFFI` identifies `Insuline` as the most important feature while `RandomForest` based models all identify `Glucose`, `DIFFI` and `EXIFFI` are instead more random. That's because `pima` has the problem of overlapping inliers and outliers and the contamination is 34%. 

Feature Selection plots similar to the ones of the other models. 

Mean Importance time → 0.14

- [x] `shuttle`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

Also here the opinion of the models is pretty different: `IF_EXIFFI`, `EIF_RandomForest` and `EIF+_RandomForest` divide the importance between Feature 0 and Feature 8 while for `DIFFI` the most important is Feature 3. `EXIFFI` is pretty random as usual. 

Feature Selection plots similar to the ones of `EIF+_RandomForest` and  `EIF_RandomForest`. 

Mean Importance time → 0.86


- [x] `wine`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

`Proline` is identified as the most important feature only in 40% of the runs, similarly to `DIFFI`. 

Feature Selection Plot similar to the one of `EXIFFI`. 

Mean Importance time → 0.078
### Synthetic Datasets

- [x] `Yaxis`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

As expected the result is the symmetric with the respect to the one obtained with `Xaxis`, random importance distribution and Feature 1 (the most important one) is identified most of the time as the least important. 

The Feature Selection plots are a bit different from the ones obtained with `Xaxis` in `scenario2`. In fact we have the Average Precision that stays constantly at 1 for the first 3 features for the blue line and for the first 4 features for the red line and then suddenly it goes down to almost 0 for the remaining features. As a result we have a sort of parallelogram as the area between the two curves. On the other hand in `scenario1` we have the usual blue line always above the red line and a big negative `AUC_FS` value. 

Mean Importance time → 0.17

- [x] `bisect`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

As expected `IF_EXIFFI` is able to identify Feature 0 and 1 as the ones sharing the importance since the artifacts of `IF` do not happen with the anomalies distributed along a bisector line. 

As expected Feature Selection plots very similar to the ones I already have in `EXIFFI`/`EXIFFI+`. 

Mean Importance time → 0.16

- [x] `bisect_6d`
	- [x] Importance Plots
		- [x] `scenario1`
		- [x] `scenario2`
	- [x] Feature Selection Plots
		- [x] `EIF+`
			- [x] `scenario1`
			- [x] `scenario2`
		- [x] `EIF`
			- [x] `scenario1`
			- [x] `scenario2`
	- [x] Local Scoremaps
		- [x] `scenario1`
		- [x] `scenario2`

As usual here the importance is evenly shared among all the features so essentially all the bars in the Bar Plot are multicolor, but there is a slight difference in the sense that `IF_EXIFFI` tends to find a single important feature. So for example in `scenario2` there is Feature 3 that has a slight advantage over the others. This effect sligthly changes also the Feature Selection plot that, in `EXIFFI`, have an increase passing from 2 to 1 features while here there is a decrease, because probably that Feature 3 is not properly the most important. 

Mean Importance time → 0.16

## Important general notes

> [!note] 
> Looking at the contamination plots:
> - `IF` was clearly the best model in `annthyroid` and in `IF_EXIFFI` the results are sligthly better then the ones of `EXIFFI`. 
> - In `moodify` `IF` is the best model (together with `DIF`) for most of the contamination values while `EIF` and `EIF+` are the worst models for most of the contamination values. 
> - In `glass` instead `IF` is a bit in the middle and its performance are sligthly better at contamination 0.1 (the level of contamination used by default for the computation of the GFI plots in all the datasets) so that may be one of the reason why there is this difference with the respect to `EXIFFI`. 
> 

> [!info] Local Scoremaps
>  The Local Scoremaps are not very interesting. In fact, similarly to the ones of `DIFFI`, the heatmap is essentially divided between a region that is completely blue and one that is completely red. In any case the `DIFFI` scoremaps were not inserted in the paper so also these ones of `IF_EXIFFI` will probably not be inserted. 

> [!info] 
> The Bar Plots of `IF_EXIFFI` differently from `EXIFFI+` tend to have less different colors in the bars, so they have less features but with higher percentages of times in a certain rank. That probably happens because the cuts performed by `IF` are along a single dimension and thus the importance is assigned to a single feature at every node while in `EIF` and `EIF+` the cuts are performed along all the dimension and so a portion of importance is assigned to each single dimension. So the explanations of `IF` tend to find a dominant feature with much higher importance score than the other and in cases in which the presence of a dominant features aligns with the ground truth (as it may be the case of `glass`) `IF_EXIFFI` is better in the explanations. Also in the Score Plot of `IF_EXIFFI` if there is a dominant feature that has a clear margin on the others in terms of importance score (e.g. in `glass`  feature `Ba` has a GFI of more than 8 while all the other features have GFI equal to 2.7 or lower). On the other hand `EXIFFI` does not create these huge differences in importance scores  (in `glass` variable `Ba`  has an importance score of 2.57 and the second one has 2.36). We can thus conclude that `EXIFFI` is better when the importance is shared across multiple features. 
## Additional Ideas

These ideas are not a must-do for the current submission but may be useful for the Rebuttal of a first review:

- Add a comparison with `AcME` if they say that we have not compared a lot of interpretation models with `ExIFFI`
- Modify the colormaps in the plots if we get a review from some expert on Data Visualization that says that the plots are not colorblind safe. 
