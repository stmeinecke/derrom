Examples:
=========

**ELPH:** Coupled electron-phonon dynamics
------------------------------------------

We utilze an auto-regressive derrom model to forecast the nonlinear transient dynamics of a coupled electron-phonon system. This example belongs to the publication

    `Data-driven forecasting of nonequilibrium solid-state dynamics <https://doi.org/10.1103/PhysRevB.107.184306`_

(can also be found on the arXiv) The ELPH folder contains the microscopic simulation code, notebooks that generate the training and testing data used in the publication, and notebooks that apply derrom models to multiple tasks. Those include the autoregression tasks, as studied in the publication, as well as some additional regression tasks


**PHELPH:** Laser model - photon-electron-phonon dynamics
---------------------------------------------------------

We utilize a derrom model to accelerate the simulation of a laser model. This is achieved by approximating the collision term, which is appears in the electron equation, by a data-driven model and dropping the explicit phonon dynamics. This example belongs to the preprint:

    `Data-Driven Acceleration of Multi-Physics Simulations <https://arxiv.org/abs/2402.16433`_
    
The folder contains the microscopic simulation code, which depends on the ELPH code; a notebook, which generates training und testing data; and notebooks that apply derrom models and implement them for the accelerated simulation.
    