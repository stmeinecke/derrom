.. derrom documentation master file, created by
   sphinx-quickstart on Fri Feb 10 13:00:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to derrom's documentation!
==================================

**derrom** is a **D**\elay **E**\mbedded **R**\egressive **R**\educed **O**\rder **M**\odel. 
It provides a modular package, which is designed to perform computationally efficient regression on high-dimensional time-series data. 
The model is trained with the supervised paradigm, where a set of existing input-output pairs is used to optimize the model parameters to obtain the desired regression performance.


What is it good for?:
---------------------

**derrom** was conceptualized to accelerate multi-phyisics simulation code. To achieve this, a trained model replaces the direct simulation of some degree's of freedom, which themselves are not of primary interested, but are required to compute the time evolution of the relevant observables.

Note that a sufficient amount of training data must, nonetheless, be generated via the potentially expensive full simulation of the system. However, if relatively few trajectories are sufficient for a well trained model and many more are to be simulated, the **derrom** approach quickly becomes worth the trouble.

A good example is the stochastic simulation of a semiconductor lasers' photon field. In that case, one likely must simulate many stochastic realizations to obtain enough samples for good statistics. The underlying charge-carrier dynamics are required for the photon field evolution and are expensive to microscopically simulate, but are not the observable of interest. Instead of resorting to rough analytic approximations, one may fully simulate the system for the first few realizations and then use the generated data to train **derrom** and use it for the remaining realizations.


How does it work:
-----------------

This is achieved by projecting the past :math:`\ell` system states (delay embedding) into a reduced dimensionality (order) latent space, which is then followed by a nonlinear transformation to obtain the feature vector for a linear regression step.



.. toctree::
   :maxdepth: 1
   :caption: Library Documentation:
   
   estimator
   dim_reducers
   scalers
   transformers
   optimizers
   utils

 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

