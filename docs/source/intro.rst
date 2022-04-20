==============
 Introduction
==============


In this project **diglm** we implement a normalizing flow hybrid model (:doc:`diglm`)
following the idea in
`this article <https://arxiv.org/1902.02767>`_ :cite:`Nalisnick2019` :
the DIGLM is a machine learning algorithm trainable in a single feed-forward
step to perform two distinct tasks, i.e.

1. Probability Density estimation
2. Classification (or any regression problem)

The first result is accomplished through the implementation of a normalizing
flow trainable function, `NeuralSplineFlow` in the :doc:`spqr` module,
with coupling layers for efficient evaluation of the Jacobian
(see :cite:`Durkan2019`).

The second task is performed with a Generalized Linear Model (GLM). The
feature vector fed to the GLM is not the intial feature vector, but they are
the "latent" features calculated by the normalizing flow. The feature
vectors used for this part of the training need to have labels, hence
the whole algorithm can be *semi-supervised* trained.


Normalizing flows and coupling layers
=====================================

A *normalizing flow* is a bijective and differentiable transformation :math:`g_{\theta}`,
which can map a vector of features with D-dimensions *x* in a transformed vector
:math:`g_\theta(x) = z` with D-dimension. Since the transformation :math:`g_\theta` is differentiable,
the probability distribution of *z* and *x* are linked through a simple change of variable:

.. math::
   p(x) = p(z) \cdot det \Bigl(\dfrac{d g_\theta}{d x}\Bigr)

Hence the name *normalizing*.
The parameters :math:`\theta` can be trained to map a simple distribution  *p(z)* (tipically a
gaussian) into the feature distribution *p(x)* through the inverse transformation
:math:`g_\theta^{-1}`.

It is possible to compose in chain bijective function to improve expressivity of the model
and speed up the computing time for the Jacobian: if at each step the bijector is made to act
as the identity on some features, the Jacobian can be made triangular (:math:`\mathcal{O}(D)` instead
of :math:`\mathcal{O}(D^3)` complexity). This is why we use coupling layers (see :cite:`Durkan2019`)
of RealNVP type to define our bijector.

GLM
===

Generalized Linear Model is a fancy name (there may be some historical reasons behind it)
for a simple "perceptron" architecture of a machine learning layer:
the inputs (features) *x* are linearly transformed applying trainable weights, and then the
output (*linear response*) is passed through an activation function to produce a *response*.
Given an objective loss, depending on labels of the inputs and the response, the weights are
trained.
Using math:

.. math::
   x \rightarrow w * x + b \rightarrow y = \dfrac{1}{1 + e^{-w*x + b}} \rightarrow loss(y_{true}, y)

In our case we had {0, 1} labeled data, therefore we choose a GLM with a sigmoid activation
function.


DIGLM
-----

The architecture of a hybrid model mixing the characteristics of normalizing flows and GLM
is simple: instead of feeding the input features to the GLM, we transform the variables to
the latent space of features *z* with the bijector and use these transformed variables as inputs
to the GLM.

The training of both the parts of the algorithm is obtained in a single feed-forward step minimizing
the log-likelihood of the labels:

.. math::
   \mathcal{L} = - \sum_i  \log{ p(y_i| x) } = - \sum_i ( \log{ p(y_i| z; \beta) } + \log{p(z; \theta) det \mathcal{J} })

where :math:`\mathcal{J}` is the Jacobian and :math:`\beta` are the GLM parameters.
We see that the minimization of this loss consists of the minimization of the objective function of both
the algorithm parts, the GLM and the normalizing flow.

As sudjested in :cite:`Nalisnick2019` we multiply the second term of the loss by a scaling constant
:math:`\lambda`, which can be tuned to allow the algorithm to train on a part more then on another,
depending on the desired performances.


References
==========
.. bibliography::
   references.bib
