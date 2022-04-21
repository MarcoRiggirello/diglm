# By Marco Riggirello and Antoine Venturini
from tensorflow_probability.python.glm import Bernoulli

from src.spqr import NeuralSplineFlow
from src.diglm import Diglm

d = Diglm(NeuralSplineFlow(splits=4), Bernoulli(), 8)

def test_features_sample_shape():
    """ Tests if Diglm can sample features with correct shape.
    """
    tnsr = d.sample()["features"]
    assert tnsr.shape.as_list() == [8]

def test_labels_sample_shape():
    """ Tests if Diglm can sample labels with correct shape.
    """
    tnsr = d.sample()["labels"]
    assert tnsr.shape.as_list() == [1]

def test_features_multiple_samples_shape_batched():
    """ Test if Diglm can multiple sample features with correct shape.
    """
    tnsr = d.sample([4,5])["features"]
    assert tnsr.shape.as_list() == [4,5,8]

def test_labels_multiple_samples_shape():
    """ Test if Diglm can multiple sample labels with correct shape.
    """
    tnsr = d.sample([4,5])["labels"]
    assert tnsr.shape.as_list() == [4,5,1]

def test_weighted_log_prob():
    """ Tests if Diglm.weighted_log_prob() has the correct shape.
    """
    tnsr = d.weighted_log_prob(d.sample())
    assert tnsr.shape.as_list() == []

def test_weighted_log_prob_batched():
    """ Tests if Diglm.weighted_log_prob() has the correct shape.
    """
    tnsr = d.weighted_log_prob(d.sample([6,3,5]))
    assert tnsr.shape.as_list() == [6,3,5]
