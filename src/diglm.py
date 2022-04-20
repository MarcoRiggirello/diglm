"""
Diglm: Deeply Invertible Generalized Linear Model
"""
from tensorflow import Variable, ones, zeros
from tensorflow_probability.python.distributions import JointDistributionNamedAutoBatched, MultivariateNormalDiag, TransformedDistribution
from tensorflow_probability.python.glm import compute_predicted_linear_response

class Diglm(JointDistributionNamedAutoBatched):
    """ Deep Invertible Generalized Linear Model using `tensorflow_probability`.
    This class implements the model described by Nalisnick et al. in 
    `Hybrid Models with Deep and Invertible Features<https://arxiv.org/abs/1902.02767>`.
    See the original article for a detailed discussion of the model, its pros and cons
    and its possible applications.
    Inherits from `tensorflow_probability.distributions.JointDistributionsNamedAutoBatched`.

    :param bijector: Bijector with learnable parameters
        for the invertible tranformation.
    :type bijector: tensorflow_probability.bijectors.Bijector
    :param glm: Generalized linear model.
    :type glm: tensorflow_probability.glm.ExponentialFamily
    :param num_feature: Dimensions of features space.
    :type num_features: int
    :param **kwargs: Other arguments for `JointDitributionNamedAutoBatched`.
    :type **kwargs: optional
    """
    def __init__(self,
                 bijector,
                 glm,
                 num_features,
                 name = "diglm",
                 **kwargs):
        self._bijector = bijector
        self._glm = glm
        self._num_features = num_features
        self._beta = Variable(ones([self.num_features]), trainable=True, name="glm_beta")
        self._beta_0 = Variable(0., trainable=True, name="glm_beta_0")
        model = {
            "features": TransformedDistribution(
                distribution=MultivariateNormalDiag(
                    loc=zeros([self.num_features]),
                    scale_diag=ones([self.num_features])
                ),
                bijector=self.bijector),
            "labels": lambda features: self.glm.as_distribution(self.eta_from_features(features))
        }
        super().__init__(model, name=name, **kwargs)

    @property
    def bijector(self):
        """ The  bijector
        """
        return self._bijector

    @property
    def glm(self):
        """ Generalized Linear Model
        """
        return self._glm

    @property
    def num_features(self):
        """ Number of features
        """
        return self._num_features

    def latent_features(self, features):
        """ Compute latent variables from features.

        :param features: Model features.
        :type features: tensorflow.Tensor
        :return: Transformed feature in latent space.
        :rtype: tensorflow.Tensor
        """
        return self.bijector.inverse(features)

    def eta_from_features(self, features):
        """ Compute predicted linear response transforming
        features in latent space.

        :param features: Model features.
        :type features: tensorflow.Tensor
        :return: Predicted linear response.
        :rtype: tensorflow.Tensor
        """
        return compute_predicted_linear_response([self.latent_features(features)],
                                                 self._beta,
                                                 offset=self._beta_0)

    def weighted_log_prob(self, value, scaling_const=.1):
        """ Weighted objective function as described in
        Nalisnik et al.

        :param value: Dictionary of (a batch of) features and labels.
        :type value: dict(tensorflow.Tensor)
        :param scaling_const: The scaling constant of the modified
            objective, defaults to `0.1`
        :type scaling_const: float
        :return: Weighted objective.
        :rtype: tensorflow.Tensor
        """
        lpp = self.log_prob_parts(value)
        return lpp["labels"] + scaling_const * lpp["features"]

    def __call__(self, features):
        """ Applies the (inverse) bijector and computes
        `mean(r)`, `var(mean)`, `d/dr mean(r)` via glm.

        :param features: Model features.
        :type features: tensorflow.Tensor
        :return: Mean, variance and derivative.
        :rtype: list[tensorflow.Tensor]
        """
        return self.glm(self.eta_from_features(features))
