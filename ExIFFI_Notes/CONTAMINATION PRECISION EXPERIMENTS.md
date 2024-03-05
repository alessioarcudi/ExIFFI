> [!note] Hyper parameters
>  - `models = [IF,EIF,EIF+,DIF]`
>  - `interpretation = [EXIFFI,DIFFI,RandomForest]`
>  - `scenario=[1,2]`

- [[CONTAMINATION PRECISION EXPERIMENTS#`EXIFFI+` and `scenario=2`|`EXIFFI+` and `scenario=2`]] → ==completed==
- [[CONTAMINATION PRECISION EXPERIMENTS#EXIFFI+` and `scenario=1`|`EXIFFI+` and `scenario=1`]] → ==completed==
- [[CONTAMINATION PRECISION EXPERIMENTS#`EXIFFI` and `scenario=2`|`EXIFFI` and `scenario=2`]]
- [[CONTAMINATION PRECISION EXPERIMENTS#`EXIFFI` and `scenario=1`|`EXIFFI` and `scenario=1`]]
- [[CONTAMINATION PRECISION EXPERIMENTS#`IF`, `DIFFI` and `scenario=2`|`IF`, `DIFFI` and `scenario=2`]]
- [[CONTAMINATION PRECISION EXPERIMENTS#`IF`, `DIFFI` and `scenario=1`|`IF`, `DIFFI` and `scenario=1`]]

## `EXIFFI+` and `scenario=2`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok

#### Real World Dataset

- `wine` → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breastw` → ok
- `ionosphere` → ok
- `annthyroid`  → ok
- `pendigits` → ok
- `diabetes` → Job 201583
- `shuttle` → Job 201579
- `moodify` → 

## `EXIFFI+` and `scenario=1`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok, put the `xlim` at 1.1
- `bisect_6d` → ok, put the `xlim` at 1.1

#### Real World Dataset

- `wine` → ==ok==
- `glass` → ==ok==
- `cardio` → ==ok==
- `pima` → ok
- `breastw` → ok, put the `xlim` at 1.1
- `ionosphere` → ok
- `annthyroid`  → ok
- `pendigits` → ok
- `diabetes` → Job 201580 
- `shuttle` → Job 201580
- `moodify` → ok, after 4h and 17 minutes of execution 

> [!note] Time per iteration
> In `diabetes` it takes 24:27 minutes on the first iteration on my pc → strangely it is faster than on CAPRI were it took 35/40 minutes per iteration (according to the `out` file of the failed jobs). Probably that has something to do with the fact that in the new implementation of `EIF_reboot` we save all the quantities in memory? 

## `EXIFFI` and `scenario=2`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok, put the `xlim` at 1.1
- `bisect_6d` → ok, put the `xlim` at 1.1

#### Real World Dataset

- `wine` → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breastw` → ok, put the `xlim` at 1.1
- `ionosphere` → ok
- `annthyroid`  → ok
- `pendigits` → ok
- `diabetes` → 
- `shuttle` → 
- `moodify` → 

## `EXIFFI` and `scenario=1`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok, put the `xlim` at 1.1
- `bisect_6d` → ok, put the `xlim` at 1.1

#### Real World Dataset

- `wine` → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breastw` → ok, put the `xlim` at 1.1
- `ionosphere` → ok
- `annthyroid`  → ok
- `pendigits` → ok
- `diabetes` → 
- `shuttle` → 
- `moodify` → 

## `IF`, `DIFFI` and `scenario=2`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok

#### Real World Dataset

- `wine` → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breastw` → ok
- `ionosphere` → ok
- `annthyroid`  → ok
- `pendigits` → ok
- `diabetes` → 
- `shuttle` → 
- `moodify` → 

## `IF`, `DIFFI` and `scenario=1`

#### Synthetic Datasets

- `Xaxis` → ok
- `Yaxis` → ok
- `bisect` → ok
- `bisect_3d` → ok
- `bisect_6d` → ok

#### Real World Dataset

- `wine` → ok
- `glass` → ok
- `cardio` → ok
- `pima` → ok
- `breastw` → ok
- `ionosphere` → ok
- `annthyroid`  → ok
- `pendigits` → ok
- `diabetes` → 
- `shuttle` → 
- `moodify` → 

# New version of `contamination` experiments 

- `wine` → ==ok==
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok 
	- `AutoEncoder` → ok
- `glass` → ==ok==
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok 
	- `AutoEncoder` → ok
- `cardio` → ==ok==
	- `EIF+` → ok
	- `EIF` → ok
	- `IF` → ok
	- `DIF` → ok 
	- `AutoEncoder` → ok

