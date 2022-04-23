===========
 Tutorials
===========

We wrote some notebooks to illustrate the functionalities of our algorithm:

1. In the :doc:`SpQR tutorial <SpQR_nb>` we demonstrate
   the functionality of our NeuralSplineFlow bijector to evaluate and
   then sample feature distributions.
2. In the :doc:`diglm tutorial <diglm_nb>`
   we train a DIGLM model on a toy dataset with
   a [0, 1] label of the feature vectors. We demonstrate the capability of
   the model to be trained both for solve the logistic regression problem and
   the evaluation of feature distribution.

.. toctree::

   SpQR_nb
   diglm_nb

   
A more complicated application
==============================

We trained the DIGLM model on a Monte Carlo `dataset simulating Beyond Standard Model events
for a heavy Higgs boson <https://archive.ics.uci.edu/ml/datasets/HIGGS>`_.

We hope you enjoy the :doc:`Higgs notebook <higgs_diglm_nb>`.


.. toctree::

   higgs_diglm_nb
