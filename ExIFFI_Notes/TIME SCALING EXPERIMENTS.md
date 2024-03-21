- [x] Put new Feature Selection Plots (obtained without scaling the data) of `Xaxis` in the paper 
	- [x] Update the $AUC_{FS}$ values of `Xaxis`
- [x] Produce the contamination plot plotting just `EIF` and`EIF+`
- [x] Finish Experiments `bisect_3d` (without rescaling) and put the result on the paper 
	- [x] Importance Plots
	- [x] Feature Selection Plots
	- [x] Contamination Plots
	- [x] Local Scoremaps
- [x] Experiments `bisect_6d` (without rescaling) and put the result on the paper 
	- [x] Performance Metrics
	- [x] Importance Plots
	- [x] Feature Selection Plots
	- [x] Contamination Plots
	- [x] Local Scoremaps
- [ ] Do Time Scaling Experiment Samples vs Dimensions
- [x] Take the `Xaxis` dataset and see how the fit predict and importances times changes as we
	- [x] Fix dimensions (e.g. 6) and increase the sample size
	- [x] Fix the sample size (e.g. 1000) and increase the number of dimensions → only 18 and 20 features missing.  

- Sample size values → 100-250-500-1000-2500-5000-10000-25000-50000-100000-250000-300000 
- Features values → 6-8-10-12-14-16-18-20 
- Importances - -Dimensioni -> stoppare oltre i 300 sec.

# Time Scaling Plots 

## Sample Size

### `Xaxis` → ==ok==

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok==
- `DIF`
	- `fit_predict` → ==ok==
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_100_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok== 
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` →  ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok==

### `Xaxis_250_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok== 
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` →  ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_500_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok==
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok==
- `DIF`
	- `fit_predict` → ==ok==
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_2500_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok==
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok==
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok==

### `Xaxis_5000_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok==
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok==
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_10000_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok== 
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_25000_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok==
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok== 
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_50000_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok== 
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok==
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_100000_6` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok== 
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok== 
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → Job 1461439 → ==ok== 

### `Xaxis_250000_6`

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → Job 1461486
	- `RandomForest` → Job 1461486
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok==
- `IF`
	- `fit_predict` → ==ok== 
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → Job on ClusterDEI
- `AE`
	- `fit_predict` → Job on ClusterDEI

### `Xaxis_300000_6`

- `EIF+`
	- `fit_predict` → 
	- `EXIFFI+` → 
- `EIF+_RF`
	- `fit_predict` → Job on ClusterDEI
	- `RandomForest` → Job on ClusterDEI
- `EIF`
	- `fit_predict` → 
	- `EXIFFI` → 
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → Job on ClusterDEI
- `AE`
	- `fit_predict` → Job on ClusterDEI

## Features 

### `Xaxis_1000_6` → ==ok== 

- `EIF+`
	- `fit_predict` → Job 1461433  → ==ok== 
	- `EXIFFI+` → Job 1461433  → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_1000_8` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok== 
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok==
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_1000_10` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok==
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok== 
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok==
- `DIF`
	- `fit_predict` → ==ok==
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_1000_12` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok==
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok==
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok==
- `DIF`
	- `fit_predict` → ==ok==
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_1000_14`

- `EIF+`
	- `fit_predict` → 
	- `EXIFFI+` → 
- `EIF+_RF`
	- `fit_predict` → 
	- `RandomForest` → 
- `EIF`
	- `fit_predict` → 
	- `EXIFFI` → 
- `IF`
	- `fit_predict` → 
	- `DIFFI` →
- `DIF`
	- `fit_predict` →
- `AE`
	- `fit_predict` → 

###  `Xaxis_1000_16`

- `EIF+`
	- `fit_predict` → 
	- `EXIFFI+` → 
- `EIF+_RF`
	- `fit_predict` → 
	- `RandomForest` → 
- `EIF`
	- `fit_predict` → 
	- `EXIFFI` → 
- `IF`
	- `fit_predict` → 
	- `DIFFI` →
- `DIF`
	- `fit_predict` →
- `AE`
	- `fit_predict` → 

###  `Xaxis_1000_18`

- `EIF+`
	- `fit_predict` → 
	- `EXIFFI+` → 
- `EIF+_RF`
	- `fit_predict` → 
	- `RandomForest` → 
- `EIF`
	- `fit_predict` → 
	- `EXIFFI` → 
- `IF`
	- `fit_predict` → 
	- `DIFFI` →
- `DIF`
	- `fit_predict` →
- `AE`
	- `fit_predict` → 

###  `Xaxis_1000_20`

- `EIF+`
	- `fit_predict` → 
	- `EXIFFI+` → 
- `EIF+_RF`
	- `fit_predict` → 
	- `RandomForest` → 
- `EIF`
	- `fit_predict` → 
	- `EXIFFI` → 
- `IF`
	- `fit_predict` → 
	- `DIFFI` →
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → 

`Xaxis_1000_40` 

- `DIF` → ==ok== 

`Xaxis_1000_80`

- `DIF` → ==ok==

The execution times are not scaling up. Also in the plots in the `DIF` paper it doesn't seem to change a lot the execution time as we increase the number of features. In any case in `DIF` they used datasets of 5000 samples and the number of features ranges between 16 and 4096 → so that we have the number of features in a logarithmic scale and we can easily represent the plot in log scale. 

Probably I should have used a similar strategy in choosing the number of samples to use for the experiments where we varied the number of samples. In the `DIF` paper they went from 1000 to 256000 samples. 

## Features new

### `Xaxis_5000_16` → ==ok==  → save these values in the pickle coming from ClusterDEI

- `EIF+`
	- `fit_predict` → ==ok== 
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok==
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_5000_32` → ==ok== 
### `Xaxis_5000_64` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok== 
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → ==ok==
	- `RandomForest` → ==ok== 
- `EIF`
	- `fit_predict` → ==ok==
	- `EXIFFI` → ==ok== 
- `IF`
	- `fit_predict` → ==ok==
	- `DIFFI` → ==ok== 
- `DIF`
	- `fit_predict` → ==ok== 
- `AE`
	- `fit_predict` → ==ok== 

### `Xaxis_5000_128` → ==ok== 

- `EIF+`
	- `fit_predict` → ==ok== 
	- `EXIFFI+` → ==ok== 
- `EIF+_RF`
	- `fit_predict` → Job 1461910 → ==ok== 
	- `RandomForest` → Job 1461910 → ==ok== 
- `EIF`
	- `fit_predict` → Job 1461910 → ==ok== 
	- `EXIFFI` → Job 1461910 → ==ok== 
- `IF`
	- `fit_predict` → Job 1461910 → ==ok== 
	- `DIFFI` → Job 1461910 → ==ok== 
- `DIF`
	- `fit_predict` → Job 1461910 → ==ok== 
- `AE`
	- `fit_predict` → Job 1461910 → ==ok== 

### `Xaxis_5000_256` → ==ok== 

- `EIF+`
	- `fit_predict` → Job 1461686 → ==ok== 
	- `EXIFFI+` → Job 1461694 → ==ok== 
- `EIF+_RF`
	- `fit_predict` → Job 1461715 → ==ok== 
	- `RandomForest` → Job 1461715 → ==ok== 
- `EIF`
	- `fit_predict` → Job 1461715 → ==ok== 
	- `EXIFFI` → Job 1461715 → ==ok== 
- `IF`
	- `fit_predict` → Job 1461715 → ==ok== 
	- `DIFFI` → Job 1461715 → ==ok== 
- `DIF`
	- `fit_predict` → Job 1461892 → ==ok== 
- `AE`
	- `fit_predict` → Job 1461892 → ==ok== 

### `Xaxis_5000_512`

- `EIF+`
	- `fit_predict` → Job 1461937 → ==ok== 
	- `EXIFFI+` → Job 1461937 → ==ok==
- `EIF+_RF`
	- `fit_predict` → Job 1462327
	- `RandomForest` → Job 1462327
- `EIF`
	- `fit_predict` → Job 1462327
	- `EXIFFI` → Job 1462327
- `IF`
	- `fit_predict` → Job 1461914 → ==ok==
	- `DIFFI` → Job 1461914 → ==ok== 
- `DIF`
	- `fit_predict` → Job 1461914 → ==ok== 
- `AE`
	- `fit_predict` → Job 1461914 → ==ok== 

### `Xaxis_5000_1024`

- `EIF+`
	- `fit_predict` → Job 1462249
	- `EXIFFI+` → Job 1462249
- `EIF+_RF`
	- `fit_predict` → Job 1462239
	- `RandomForest` → Job 1462239
- `EIF`
	- `fit_predict` → Job 1462239
	- `EXIFFI` → Job 1462239
- `IF`
	- `fit_predict` → Job 1461960 → ==ok==
	- `DIFFI` → Job 1461960 → ==ok== 
- `DIF`
	- `fit_predict` → Job 1461960 → ==ok== 
- `AE`
	- `fit_predict` → Job 1462239

### `Xaxis_5000_2048`

- `EIF+`
	- `fit_predict` → 
	- `EXIFFI+` → 
- `EIF+_RF`
	- `fit_predict` → 
	- `RandomForest` → 
- `EIF`
	- `fit_predict` → 
	- `EXIFFI` → 
- `IF`
	- `fit_predict` → 
	- `DIFFI` →
- `DIF`
	- `fit_predict` →
- `AE`
	- `fit_predict` → 

### `Xaxis_5000_4096`

- `EIF+`
	- `fit_predict` → 
	- `EXIFFI+` → 
- `EIF+_RF`
	- `fit_predict` → 
	- `RandomForest` → 
- `EIF`
	- `fit_predict` → 
	- `EXIFFI` → 
- `IF`
	- `fit_predict` → 
	- `DIFFI` →
- `DIF`
	- `fit_predict` →
- `AE`
	- `fit_predict` → 
