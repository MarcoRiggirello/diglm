# Marco Riggirello & Antoine Venturini
from tensorflow import expand_dims, reshape, Module
from tensorflow.python.keras.layers import Layer, Dense, Activation
from tensorflow.python.keras.activations import softmax, softplus
from tensorflow_probability.python.bijectors import RealNVP, Chain, RationalQuadraticSpline


class SplineBlock(Layer):
    """ SplineBlock class
    """
    def __init__(self,
                 nunits,
                 nbins,
                 border,
                 hidden_layers=[512,512],
                 min_bin_gap=1e-3,
                 min_slope=1e-3):
        super().__init__(name="spline_block")
        self._nunits = nunits
        self._nbins = nbins
        self._nslopes = nbins - 1
        self._border = border
        self._min_bin_gap = min_bin_gap
        self._min_slope = min_slope
        self._hidden_layers = [
            Dense(n,activation="relu",name=f"spqr_nn_layer_{i}")
            for i,n in enumerate(hidden_layers)
        ]
        self._widths_layer = Dense(self._nunits * self._nbins, name="widths_layer")
        self._heights_layer = Dense(self._nunits * self._nbins, name="heights_layer")
        self._slopes_layer = Dense(self._nunits * self._nslopes, name="slopes_layer")

    def call(self, units):
        if units.shape.rank == 1:
            units = expand_dims(units, axis=0)
            adjust_rank = lambda x: x[0]
        else:
            adjust_rank = lambda x: x
        for layer in self._hidden_layers:
            units = layer(units)

        widths = adjust_rank(self._widths_layer(units))
        widths = reshape(widths,
                         widths
                         .shape[:-1]
                         .concatenate((self._nunits, self._nbins)))
        widths = Activation(softmax)(widths)
        widths = widths * (2 * self._border - self._nbins * self._min_bin_gap ) - self._min_bin_gap

        heights = adjust_rank(self._heights_layer(units))
        heights = reshape(heights,
                          heights
                          .shape[:-1]
                          .concatenate((self._nunits, self._nbins)))
        heights = Activation(softmax)(heights)
        heights = heights * (2 * self._border - self._nbins * self._min_bin_gap ) - self._min_bin_gap

        slopes = adjust_rank(self._slopes_layer(units))
        slopes = reshape(slopes,
                         slopes
                         .shape[:-1]
                         .concatenate((self._nunits, self._nslopes)))
        slopes = Activation(softplus)(slopes)
        slopes = slopes + self._min_slope

        return widths, heights, slopes


class SplineInitializer(Module):
    """ SplineInitializer class
    """
    def __init__(self,
                 nbins=128,
                 border=4,
                 hidden_layers=[512,512],
                 min_bin_gap=1e-3,
                 min_slope=1e-3):
        super().__init__()
        self._nbins = nbins
        self._border = border
        self._min_bin_gap = min_bin_gap
        self._min_slope = min_slope
        self._hidden_layers = hidden_layers
        self._built = False

    def __call__(self,
                 x,
                 nunits):
        if not self._built:
            self._nn = SplineBlock(nunits,
                                   self._nbins,
                                   self._border,
                                   hidden_layers=self._hidden_layers,
                                   min_bin_gap=self._min_bin_gap,
                                   min_slope=self._min_slope)
            self._built = True
        widths, heights, slopes = self._nn(x)
        return RationalQuadraticSpline(widths,
                                       heights,
                                       slopes,
                                       range_min= -self._border)

class NeuralSplineFlow(Chain):
    """ Neural Spline Flow bijector.
    """
    def __init__(self,
                 splits=None,
                 masks=None,
                 spline_params = {}
                 ):

        self._spline_params = spline_params
        self._splits = splits
        self._masks = masks
        if self._splits is not None and self._masks is None:
            if self._splits < 2:
                raise ValueError("splits must be greater than or equal to 2 ",
                                 "(You must split your feature vec in at least two parts).")
            realnvp_args = [
                dict(fraction_masked=i/self._splits, bijector_fn=SplineInitializer(**self._spline_params))
                for i in range(1-self._splits, self._splits) if i != 0
            ]
        elif self._masks is not None and self._splits is None:
            realnvp_args = [
                dict(num_masked=i, bijector_fn=SplineInitializer(**self._spline_params))
                for i in self._masks
            ]
        else:
            raise ValueError("You must specify `splits` OR `masks`, not both.")

        self._coupling_layers = [
            RealNVP(**splines, name=f"coupling_layer_{i}")
            for i, splines in enumerate(realnvp_args)
        ]
        super().__init__(bijectors=self._coupling_layers, name="nsf")

