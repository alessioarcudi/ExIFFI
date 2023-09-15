# ExIFFI

The Python script and Jupyter Notebooks composing this Repository contian the code used to produce the results presente in the "ExIFFI and EIF+: Interpretability and Enhanced Generalizability to Extend the
Extended Isolation Forest" paper. 

## Repository Organization 

- models -> This folder contains the Python scripts defining the classes implementing the models presented in the paper: Extended Isolation Forest (EIF), Extended Isolation Forest Plus (EIF+) and ExIFFI: Extended Isolation Forest Feature Importance. The script called forests.py contains a more efficient implementation of the EIF model.

- notebooks -> This folder contains some Jupyter Notebooks that were used to obtain the graphical and numerical results presented in the paper.

- data -> Here the pkl, csv and mat files, containing the synthetic and real-world dataset used for the experiments, can be found. 

- utils -> Finally, this folder contains some Pyhton scripts were utility functions, used in the Notebooks and in the model classes, are defined. 
