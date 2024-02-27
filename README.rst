Delay Embedded Regressive Reduced Order Model (derrom)
======================================================

**derrom** is a **D**\elay **E**\mbedded **R**\egressive **R**\educed **O**\rder **M**\odel. 
It provides a modular package, which is designed to perform computationally efficient regression on high-dimensional time-series data. 
The model is trained with the supervised paradigm, where a set of existing input-output pairs is used to optimize the model parameters to obtain the desired regression performance.


Library Reference
-----------------

is hosted by readthedocs:
    
    https://derrom.readthedocs.io/en/latest/


What is it good for?:
---------------------

**derrom** was conceptualized to accelerate multi-physics simulation code. To achieve this, a trained model replaces the direct simulation of some degree's of freedom, which themselves are not of primary interested, but are required to compute the time evolution of the relevant observables. Moreover, the delay embedding permits to omit the degree's of freedom, which do not couple to the considered observables, altogether, but keep their effect in the trained model.

A sufficient amount of training data must, nonetheless, be generated via the potentially expensive full simulation of the system. However, if relatively few trajectories are sufficient for a well trained model and many more are to be simulated, the **derrom** approach quickly becomes worth the trouble.

A good example is the stochastic simulation of a semiconductor lasers' photon field. In that case, one likely must simulate many stochastic realizations to obtain enough samples for good statistics. The underlying charge-carrier dynamics are required for the photon field evolution and are expensive to microscopically simulate, but are not the observable of interest. Instead of resorting to rough analytic approximations, one may fully simulate the system for the first few realizations and then use the generated data to train **derrom** and use it for the remaining realizations.


Who is it for?:
---------------

The derrom package is intended for research/scientific use. Its focus is on easy experimentation, which is achieved via a modular implementation. For instance, further dimensionality reduction approaches and nonlinear transformations can be quickly integrated via additional moduls.

Note that the code is not optimized for absolute computational speed and minimal memory usage. We assume, that once a well working derrom model is obtained, it would be integrated into an existing simulation code by reimplementing it within the code's framework.

How does it work?:
------------------

**derrom** first projects the past :math:`\ell` system states (delay embedding) into a reduced dimensionality (order) latent space. This step is designed retain the dominant patterns of the trajectories and get rid of redundant and irrelevant information. This both mitigates the curse of dimensionality and promotes robust regression. Next, the latent space features are scaled to appropriate magnitudes. Then, the past :math:`\ell` scaled reduced system states are subject to a nonlinear transformation, which yields **derrom**\'s feature vector. Lastly, the regression step is taken via a linear map, i.e., a matrix multiplication.

How to Cite:
------------

If you want to use derrom in the scientific context, please do not forget to provide a citation:
https://doi.org/10.1103/PhysRevB.107.184306

Installation:
-------------

Use pip::

	pip install derrom

If you want the example notebooks and some experimental modules: clone the repository::

    git clone https://github.com/stmeinecke/derrom.git
    
Make sure you fulfill all requirements, e.g., via::

    pip install -r requirements.txt

and you're ready to go!


Examples:
---------

A number of examples, which currently include autoregressive forecasting of nonlinear solid-state transient dyanamics and the acceleration of a multi-physics laser simulation, are provided here:
    
    https://github.com/stmeinecke/derrom/tree/main/examples

