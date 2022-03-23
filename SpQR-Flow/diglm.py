from tensorflow import Module, Variable, ones, zeros
from tensorflow_probability.python.distributions import JointDistributionNamedAutoBatched, MultivariateNormalDiag, TransformedDistribution
from tensorflow_probability.python.glm import compute_predicted_linear_response


class DIGLM(Module):
    """ Deep Invertible Generalized Linear Model
    """
    def __init__(self,
                 bijector,
                 glm,
                 num_features,
                 ):
        super().__init__()
        self._bijector = bijector
        self._glm = glm
        self._num_features = num_features
        self._beta = Variable(ones([self.num_features]), trainable=True)
        self._beta_0 = Variable(0., trainable=True)
        self._latent_distribution = MultivariateNormalDiag(loc=zeros([self.num_features]),
                                                           scale_diag=ones([self.num_features]))
        self._joint_distribution = JointDistributionNamedAutoBatched(
            {
                "x": TransformedDistribution(distribution=self.latent_distribution,
                                             bijector=self.bijector),
                "y": lambda x: self.glm.as_distribution(self.eta_from_features(x))
            }
        )

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

    @property
    def latent_distribution(self):
        """ Return a tfd.Distribution representing the latent distribution
        (a multivariate normal diagonal distribution).
        """
        return self._latent_distribution

    @property
    def joint_distribution(self):
        """ Returns the features distribution as tfd.Distribution.
        """
        return self._joint_distribution

    def latent_features(self, features):
        """ Compute latent variables from features.
        """
        return self.bijector.forward(features)

    def eta_from_features(self, features):
        """ Compute predicted linear response transforming
        features in latent space.
        """
        return compute_predicted_linear_response([self.latent_features(features)],
                                                 self._beta,
                                                 offset=self._beta_0)
