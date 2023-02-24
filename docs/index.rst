.. derrom documentation master file, created by
   sphinx-quickstart on Fri Feb 10 13:00:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to derrom's documentation!
==================================

This library provides a **D**\elay **E**\mbedded **R**\egressive **R**\educed **O**\rder **M**\odel. 


The model is designed to perform computationally efficient regression on high-dimensional time-series data. 
This is achieved by projecting the past :math:`\ell` system states (delay embedding) into a reduced dimensionality (order) latent space, which is then followed by a nonlinear transformation to obtain the feature vector for the linear regression step.

The model is trained with the supervised paradigm

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

