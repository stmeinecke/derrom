.. derrom documentation master file, created by
   sphinx-quickstart on Fri Feb 10 13:00:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to derrom's documentation!
==================================

**derrom** is a **D**\elay **E**\mbedded **R**\egressive **R**\educed **O**\rder **M**\odel. 
It provides a modular package, which is designed to perform computationally efficient regression on high-dimensional time-series data. 
The model is trained with the supervised paradigm, where a set of existing input-output pairs is used to optimize the model parameters to obtain the desired regression performance.

.. toctree::
   :maxdepth: 1
   :caption: Library Documentation:
   
   estimator
   dim_reducers
   scalers
   transformers
   optimizers
   utils


What is it good for?:
---------------------

**derrom** was conceptualized to accelerate multi-phyisics simulation code. To achieve this, a trained model replaces the direct simulation of some degree's of freedom, which themselves are not of primary interested, but are required to compute the time evolution of the relevant observables. Moreover, the delay embedding permits to omit the degree's of freedom, which do not couple to the considered observables, altogether, but keep their effect in the trained model.

Note that a sufficient amount of training data must, nonetheless, be generated via the potentially expensive full simulation of the system. However, if relatively few trajectories are sufficient for a well trained model and many more are to be simulated, the **derrom** approach quickly becomes worth the trouble.

A good example is the stochastic simulation of a semiconductor lasers' photon field. In that case, one likely must simulate many stochastic realizations to obtain enough samples for good statistics. The underlying charge-carrier dynamics are required for the photon field evolution and are expensive to microscopically simulate, but are not the observable of interest. Instead of resorting to rough analytic approximations, one may fully simulate the system for the first few realizations and then use the generated data to train **derrom** and use it for the remaining realizations.


How does it work?:
------------------

**derrom** first projects the past :math:`\ell` system states (delay embedding) into a reduced dimensionality (order) latent space. This step is designed retain the dominant patterns of the trajectories and get rid of redundant and irrelevant information. This both mitigates the curse of dimensionality and promotes robust regression. Next, the latent space features are scaled to appropriate magnitudes. Then, the past :math:`\ell` scaled reduced system states are subject to a nonlinear transformation, which yields **derrom**\'s feature vector. Lastly, the regression step is taken via a linear map, i.e., a matrix multiplication.


Installation:
-------------

Clone the repository::

    git clone https://github.com/stmeinecke/derrom.git
    
Make sure you fulfill all requirements and you're ready to go!


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

