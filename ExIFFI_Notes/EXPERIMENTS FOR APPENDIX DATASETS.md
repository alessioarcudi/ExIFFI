# `wine` → ==ok== 

- [x] Feature Selection Plots 
	- [x] Put on paper
- [x] Contamination Plots 
	- [x] Put on paper
- [x] Performance Metrics Table 
	- [x] Write it on the paper
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps
	- [x] Put on paper

# `breastw` → ==ok== 

- [x] Feature Selection Plots 
	- [x] Change position of the $AUC_{FS}$ box in a notebook for the plots of scenario 1
	- [x] Put on paper 
- [x] Contamination Plots 
	- [x] Put on paper 
- [x] Performance Metrics Table 
	- [x] Put on paper 
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps
	- [x] Put on paper

# `pima` → ==ok== 

- [x] Feature Selection Plots 
	- [x] Put on paper 
- [x] Contamination Plots 
	- [x] Put on paper 
- [x] Performance Metrics Table 
	- [x] Put on paper 
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps
	- [x] Put on paper

# `cardio` → ==ok== 

- [x] Feature Selection Plots 
	- [x] Put on paper 
- [x] Contamination Plots 
	- [x] Put on paper 
- [x] Performance Metrics Table 
	- [x] Time Scaling experiment on importances of `RandomForest`
	- [x] Put on paper 
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps
	- [x] Put on paper

List of feature names for `cardio` (found [here](https://archive.ics.uci.edu/dataset/193/cardiotocography)):

- LB - FHR baseline (beats per minute)
- AC - # of accelerations per second
- FM - # of fetal movements per second
- UC - # of uterine contractions per second
- DL - # of light decelerations per second
- DS - # of severe decelerations per second
- DP - # of prolongued decelerations per second
- ASTV - percentage of time with abnormal short term variability
- MSTV - mean value of short term variability
- ALTV - percentage of time with abnormal long term variability
- MLTV - mean value of long term variability
- Width - width of FHR histogram
- Min - minimum of FHR histogram
- Max - Maximum of FHR histogram
- Nmax - # of histogram peaks
- Nzeros - # of histogram zeros
- Mode - histogram mode
- Mean - histogram mean
- Median - histogram median
- Variance - histogram variance
- Tendency - histogram tendency
- CLASS - FHR pattern class code (1 to 10)
# `ionosphere` → ==ok== 

- [x] Feature Selection Plots → The execution blocks in `EIF_EIF_RandomForest` and `EIF_IF_RandomForest` for a `ZeroDivisionError`.
	- [x] Missing:
		- [x] `random` scenario 2 both in `EIF` and `EIF+`
		- [x] `IF_DIFFI` scenario 2 both in `EIF` and `EIF+`
		- [x] `IF_RandomForest` scenario 2 both in `EIF` and `EIF+`
		- [x] `EIF_RandomForest` scenario 2 both in `EIF` and `EIF+`
	- [x] Put on paper 
- [x] Contamination Plots 
	- [x] Put on paper 
- [x] Performance Metrics Table 
	- [x] Time Scaling experiment on importances of `RandomForest`
	- [x] Put on paper 
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps
	- [x] Put on paper
# `pendigits` → ==ok== 

- [x] Feature Selection Plots 
	- [x] Put on paper 
- [x] Contamination Plots 
	- [x] Put on paper 
- [x] Performance Metrics Table 
	- [x] Time Scaling experiment on importances of `RandomForest`
	- [x] Put on paper 
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps
	- [x] Put on paper
# `shuttle` → ==ok== 

- [x] Feature Selection Plots 
	- [x] Put on paper 
- [x] Contamination Plots 
	- [x] Put on paper 
- [x] Performance Metrics Table 
	- [x] Time Scaling experiment on importances of `RandomForest`
	- [x] Put on paper 
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps 
	- [x] Put on papertab:fs-tab-Wine

> [!note] 
>  The scales in the Local Scoremaps are different from the ones used in the previous version of the paper because in that case we down sampled big datasets to 2500 samples, now instead we downsample to 7500 samples so the scales are obviously different. Actually since we are taking more samples from the complete dataset the scales will be similar to the scales of the entire dataset. The same holds for `moodify`. 
# `diabetes` → ==ok== 

- [x] Feature Selection Plots 
	- [x] Put on paper 
- [x] Contamination Plots 
	- [x] Put on paper 
- [x] Performance Metrics Table 
	- [x] Time Scaling experiment on importances of `RandomForest`
	- [x] Put on paper 
- [x] Importance Plots 
	- [x] Put on paper 
- [x] Local Scoremaps
	- [x] Put on paper