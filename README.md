# comboLTR: Modeling drug combination effects via latent tensor reconstruction

## Overview

comboLTR is a new polynomial regression-based framework for modeling anti-cancer effects of drug combinations in various doses and across different cancer cell lines. It is implemented in Python. 

The data used in the experiments is available on: https://doi.org/10.5281/zenodo.4625084.

## Instructions

*ltr_tensor_solver_actxu_v_cls_010.py* This file contains the code for polynomial regression via latent tensor reconstruction.

The description of the interface of the solver is given in readme.pdf file.


*comboLTR_CV.py* This file contains the code for cross validations on the full dataset used in the paper.

## Dependencies

- numpy
- scikit-learn
- scipy

## Citing comboLTR

