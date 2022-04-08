.. SpQR-Flow documentation master file, created by
   sphinx-quickstart on Tue Apr  5 12:08:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SpQR-Flow's documentation!
=====================================

**SpQR-Flow** is a Python library where a normalizing flow hybrid model is
implemented, following the idea in
`this article <https://arxiv.org/pdf/1902.02767.pdf/>`_: 
the built normalizing flow parameters are trained with `keras.layer.Dense`
and a Machine Learning algorithm to solve the *two* following problems:

1. Probability Density estimation
2. Classification


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::

   api
   plot_utils_api
   download_api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
