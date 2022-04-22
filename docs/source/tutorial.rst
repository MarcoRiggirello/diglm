=========
Tutorials
=========

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
for a heavy Higgs boson <https://archive.ics.uci.edu/ml/datasets/HIGGS>`_.

We hope you enjoy the :doc:`Higgs notebook <notebooks/diglm_nb>`.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

Notebooks list
--------------

.. toctree::

   notebooks/SpQR_nb
   notebooks/diglm_nb
   notebooks/higgs_diglm_nb
