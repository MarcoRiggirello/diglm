# Author: Marco Riggirello
#==================================================================================
""" Various utilities
"""


import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras.activations import softmax, softplus
from tensorflow_probability.bijectors import RationalQuadraticSpline


class SplineBlock(tfkl.Layer):
    """ SplineBlock class
    """
    def __init__(self,
                 hidden_layers=[32, 64, 64],
                 **kwargs):
        super().__init__(**kwargs)
        self._layers = []
        for i,n in enumerate(hidden_layers):
            self._layers.append(tfkl.Dense(n,
                                           activation="relu",
                                           name=f"spqr_nn_layer_{i}"))

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class BinsLayer(tfkl.Layer):
    """ BinsLayer class
    """
    def __init__(self,
                 nunits,
                 nbins,
                 border,
                 min_bin_gap=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self._nunits = nunits
        self._nbins = nbins
        self._border = border
        self._min_bin_gap = min_bin_gap
        self._dense = tfkl.Dense(self._nunits * self._nbins,
                                 activation=softmax)
        self._reshape = tfkl.Reshape((-1, self._nunits * self._nbins))

    def call(self, x):
        x = self._dense(x)
        x = self._reshape(x)
        return x * (2 * self._border - self._nbins * self._min_bin_gap ) - self._min_bin_gap


class SlopesLayer(tfkl.Layer):
    """ SlopesLayer class
    """
    def __init__(self,
                 nunits,
                 nbins,
                 min_slope=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self._nunits = nunits
        self._nslopes = nbins - 1
        self._min_slope = min_slope
        self._dense = tfkl.Dense( self._nunits * self._nslopes, activation=softplus)
        self._reshape = tfkl.Reshape((-1, self._nunits * self._nslopes))

    def call(self, x):
        x = self._dense(x)
        x = self._reshape(x)
        return x + self._min_slope


class SplineInitializer(tf.Module):
    """ SplineInitializer class
    """
    def __init__(self,
                 nbins=128,
                 border=4,
                 nn=SplineBlock(),
                 min_bin_gap=1e-3,
                 min_slope=1e-3):
        super().__init__()
        self._nbins = nbins
        self._border = border
        self._nn = nn
        self._min_bin_gap = min_bin_gap
        self._min_slope = min_slope
        self._built = False
        self._bin_widths = None
        self._bin_heights = None
        self._knot_slopes = None

    @tf.function
    def __call__(self,
                 x,
                 nunits):
        if not self._built:
            self._bin_widths = BinsLayer(nunits,
                                         self._nbins,
                                         self._border,
                                         min_bin_gap=self._min_bin_gap,
                                         name="w" )
            self._bin_heights = BinsLayer(nunits,
                                          self._nbins,
                                          self._border,
                                          min_bin_gap=self._min_bin_gap,
                                          name="h" )
            self._knot_slopes = SlopesLayer(nunits,
                                            self._nbins,
                                            min_slope=self._min_slope,
                                            name="s" )
            self._built = True
        if self._nn is not None:
            x = self._nn(x)
        return RationalQuadraticSpline(self._bin_widths(x),
                                       self._bin_heights(x),
                                       self._knot_slopes(x),
                                       range_min= -self._border)
