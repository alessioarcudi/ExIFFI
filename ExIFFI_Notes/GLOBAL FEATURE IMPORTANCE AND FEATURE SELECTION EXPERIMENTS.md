
# Experiment Configurations

- `models = [IF,EIF,EIF+]`
- `interpretation = [EXIFFI,DIFFI,RandomForest]`
- `scenario=[1,2]`

## `EIF+` , `EXIFFI` and `scenario=2`

### Synthetic Dataset

- `Xaxis`
	- Global Importance 40 run no pre process → `22-02-2024_22-30` → this makes sense
	- Feature Importance → `22-02-2024_22-32` → makes sense 
	- Global Importance 40 run pre process → `22-02-2024_22-44` → very similar to no pre process (probably because the training data are centered on the origin and so they are already pre processed )
	- Feature Importance → `22-02-2024_22-47` → makes sense 
- `Yaxis`:
	- Global Importance 40 run pre process → `22-02-2024_22-49` → ok
	- Feature Selection → `22-02-2024_22-50` → ok
- `bisect`:
	- Global Importance 40 run pre process → `22-02-2024_22-53` → there is a little piece of green (but that's probably because we used 40 runs and so there is the chance of getting the unlucky one)
	- Feature Selection → `22-02-2024_22-54` → ok
- `bisect_3d`:
	- Global Importance 40 run pre process → `22-02-2024_22-56` → ok
	- Feature Selection → `22-02-2024_22-56` → ok → sligthly worse than `bisect` but that's because there are more anomalous features and so the interpretation task is more difficult 
- `bisect_6d`:
	- Global Importance 40 run pre process → `22-02-2024_22-56` → ok
	- Feature Selection → `22-02-2024_22-50` → ok

### Real World Dataset

- `wine` :
	- Global Importance 10 run → `22-02-2024_21-38`
	- Feature Selection → `22-02-2024_21-53` → no sense → maybe it's the dataset that is bad 
	- Global Importance 40 run → `22-02-2024_22-02`
	- Feature Importance → `22-02-2024_22-07`
- `glass`
- `cardio`
	- Global Importance 40 run → `22-02-2024_23-23`
	- Feature Selection → 
- `pima`
- `breastw`
- `ionosphere`
- `annthyroid`
- `pendigits`
- `diabetes`
- `shuttle`
- `moodify`

## `EIF+` , `EXIFFI` and `scenario=1`

- `wine` :
	- Global Importance 10 run → `22-02-2024_22-14`
	- Feature Selection → `22-02-2024_22-16` → no sense 

## `EIF` , `EXIFFI` and `scenario=2`

- `wine` :
	- Global Importance 40 run pre process → `22-02-2024_22-37` → with pre process much better than without pre process
	- Feature Selection → `22-02-2024_22-38` → no sense 

Risultati primi test

- Sui dataset sintetici i risultati sono quelli attesi (a parte `bisect_6d` che non so per quale strano motivo non mi trova nella cartella). La Feature Selection credo sia nella sua forma migliore possibile → rettangolo che copre quasi tutto il plot. Diventa un pò peggio con `bisect_3d` ma credo sia perchè ci sono più dimensioni su cui si distribuiscono le anomalie e quindi è più difficile per il modello dare un interpretazione precisa. 
- Su `wine` i risultati non sono proprio bellissimi → nella Feature Selection la parte `inverse` sale si ma parte da valori bassi quindi la figura che viene fuori è una sorta di triangolo 
- Su `cardio` la situazione è abbastanza particolare. Il plot della Feature Selection ha la forma di una balena direi. La parte `inverse` sta stazionaria poi scende da 5 feature a 2 e poi fa un balzo in alto da 2 a 1 feature. Dall'altra parte la parte `direct` scende ma non cosi repentinamente come in `wine` e in quelli sintetici e poi sale all'ultimo 
- Non so per quale motivo ma su `pima` e `breastw` mi da un errore `split_dataset()` (sembra che non ci siano inlier ma ho già ricontrollato e non so che problemi abbia) → non ho controllato su tutti i dataset quindi `pima` e `breastw` potrebbero non essere gli unici con sto errore. 