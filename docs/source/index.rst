.. SpQR-Flow documentation master file, created by
   sphinx-quickstart on Tue Apr  5 12:08:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to diglm's documentation!
=================================

**diglm** is the name of a project developed by Marco Riggirello and
Antoine Venturini for an exam on Computing Methods for Experimental Particle
Physics.

In this project we implement a normalizing flow hybrid model (DIGLM)
following the idea in
`this article <https://arxiv.org/1902.02767>`_ :cite:`Nalisnick2019` :
the DIGLM is a machine learning algorithm trainable in a single feed-forward
step to perform two distinct tasks, i.e.

1. Probability Density estimation
2. Classification (or any regression problem)

The first result is accomplished through the implementation of a normalizing
flow trainable function with coupling layers for efficient evaluation of
the Jacobian (see :cite:`Durkan2019`). The class `NeuralSplineFlow`
implements this part of the algorithm.

The second task is performed with a Generalized Linear Model (GLM). The
feature vector fed to the GLM is not the intial feature vector, but they are
the "latent" features calculated by the normalizing flow. The feature
vectors used for this part of the training need to have labels, hence
the whole algorithm can be *semi-supervised* trained.


.. bibliography::
   references.bib

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::

   api
   plot_utils_api
   download_api
   notebooks/SpQR_nb
   notebooks/diglm_nb
   notebooks/higgs_uci_nb
   notebooks/higgs_diglm_nb
   
Tutorials
---------

We wrote some notebooks to illustrate the functionalities of our algorithm:

1. In the :doc:`SpQR tutorial <notebooks/SpQR_nb>` we demonstrate
   the functionality of our NeuralSplineFlow bijector to evaluate and
   then sample feature distributions.
2. In the :doc:`diglm tutorial <notebooks/diglm_nb>`
   we train a DIGLM model on a toy dataset with
   a [0, 1] label of the feature vectors. We demonstrate the capability of
   the model to be trained both for solve the logistic regression problem and
   the evaluation of feature distribution.

A more complicated application
------------------------------

We trained the DIGLM model on a Monte Carlo `dataset simulating Beyond Standard Model events
for a heavy Higgs boson <https://archive.ics.uci.edu/ml/datasets/HIGGS>_`.

Follow the link to the :doc:`Higgs notebook <notebooks/diglm_nb>`.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
