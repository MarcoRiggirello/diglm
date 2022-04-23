==================
Code documentation
==================

The hybrid model architecture is built with extensive use of tools and classes
from the ``TensorFlow`` ecosystem.

The model is built in two steps:

#. In the :doc:`spqr` module we define the class ``NeuralSplineFlow``,
   which creates the bijector with Rational Quadratic Spline and RealNVP
   coupling layers. This part of the codes allows to define a trainable
   :term:`normalizing flow` model;
#. The :doc:`diglm` class builds the complete architecture, chaining together
   the bijector in ``spqr`` and a :term:`GLM` layer.


DIGLM model
===========

.. toctree::
   :maxdepth: 1

   spqr
   diglm


Helper modules
==============

We have written some functions to help us with the various tutorials.
They allows to make gifs and download datasets from the internet.

.. toctree::
   :maxdepth: 1

   plot_utils_api
   download_api


API
===

.. toctree::
   :maxdepth: 1

   api
	    
