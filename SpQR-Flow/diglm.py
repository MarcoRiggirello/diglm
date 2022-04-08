from tensorflow import Variable, ones, zeros
from tensorflow_probability.python.distributions import JointDistributionNamedAutoBatched, MultivariateNormalDiag, TransformedDistribution
from tensorflow_probability.python.glm import compute_predicted_linear_response


__version__ = "0.1"

class DIGLM(JointDistributionNamedAutoBatched):
    """ Deep Invertible Generalized Linear Model
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
        self._beta = Variable(ones([self.num_features]), trainable=True)
        self._beta_0 = Variable(0., trainable=True)
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
        """
        return self.bijector.inverse(features)

    def eta_from_features(self, features):
        """ Compute predicted linear response transforming
        features in latent space.
        """
        return compute_predicted_linear_response([self.latent_features(features)],
                                                 self._beta,
                                                 offset=self._beta_0)

    def weighted_log_prob(self, value, scaling_const=.1):
        lpp = self.log_prob_parts(value)
        return lpp["labels"] + scaling_const * lpp["features"]

    def __call__(self, features):
        """ Applies the (inverse) bijector and computes
        `mean(r)`, `var(mean)`, `d/dr mean(r)` via glm.
        """
        return self.glm(self.eta_from_features(features))
