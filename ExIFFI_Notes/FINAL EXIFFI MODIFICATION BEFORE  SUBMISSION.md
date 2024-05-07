
Final little modifications of `ExIFFI` before new submission to `EAAI` Journal. 

- [ ] Summarize the paper
	- [ ] Remove `DIFFI` explanation → not necessary
	- [ ] Rephrase the role of `ExIFFI` as an Interpretability method that work for different versions of Isolation Forest (i.e. `IF`, `EIF` and `EIF+`)
	- [ ] Move one (or more) datasets from the `Experimental Results` sections to the `Appendix` (e.g `Bisect6D`) to reduce the number of pages and making the `Experimental Results` section less redundant and repetitive
	- [ ] Summarize some explanations. For example with the new Overleaf template of `EAAI` new the Local Scoremaps description is one page long → too much. 
- [ ] Additional Experiments
	- [ ] To prove the usability of `ExIFFI` as an Interpretability method for `IF` based models we need to try to see what happens if we use it to interpret the `IF` model. So we have to use the `IsolationForest` class (contained in `EIF_reboot.py`) as the AD model to explain. 
		- [ ] Produce the Bar and Score Plot 
		- [ ] Produce the Feature Selection Plot and compute the new `AUC_FS` values
		- [ ] Produce the Local Scoremaps 
	- [ ] Add the new produced Feature Selection Plot in the paper
	- [ ] Try to compute the GFI scores again using as contamination the contamination of the dataset (i.e. `dataset.perc_outliers`) instead of the default value of 0.1. I hope this does not change much otherwise I should redo all the experiments for all the datasets. 

## Experiments

Keep track of the experiments on `IF_EXIFFI`. Let's starts first with the datasets included in the main part of the paper. 

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

### Important general notes

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