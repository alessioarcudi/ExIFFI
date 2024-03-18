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

