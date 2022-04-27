# By Marco Riggirello and Antoine Venturini
import numpy as np
import tensorflow as tf

from src import spqr

def test_coupling_layer_identity_positive_mask():
    x = tf.ones([12])
    nsf = spqr.NeuralSplineFlow([4])
    y = nsf.forward(x)
    assert np.all(x[:4] == y[:4])

def test_coupling_layer_identity_negative_mask():
    x = tf.ones([12])
    nsf = spqr.NeuralSplineFlow([-4])
    y = nsf.forward(x)
    assert np.all(x[-4:] == y[-4:])

def test_bijector_bijectivity_splits():
    x = tf.ones([12])
    nsf = spqr.NeuralSplineFlow.from_split_features(4,12)
    y = nsf.forward(x)
    assert np.all(x  == nsf.inverse(y))

def test_bijector_bijectivity_masks():
    x = tf.ones([12])
    nsf = spqr.NeuralSplineFlow([4,3,-2,-4])
    y = nsf.forward(x)
    assert np.all(x == nsf.inverse(y))
