# Marco Riggirello & Antoine Venturini
from tensorflow import concat, expand_dims, function, reshape, shape, Module, Variable
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.activations import softmax, softplus
from tensorflow_probability.python.bijectors import RealNVP, Chain, RationalQuadraticSpline

class RobustDense(Layer):
    """ Dense layer not prone to rank 1 input problem.
    """
    def __init__(self, nunits, *args, **kwargs):
        super().__init__()
        self._dense = Dense(nunits, *args, **kwargs)

    def call(self, units):
        if units.shape.rank == 1:
            units = expand_dims(units, axis=0)
            reshape_output = lambda x: x[0]
        else:
            reshape_output = lambda x: x
        units = self._dense(units)
        return reshape_output(units)


class SplineBlock(Layer):
    """ SplineBlock class
    """
    def __init__(self,
                 hidden_layers=[512,512]):
        super().__init__()
        self._layers = [
            RobustDense(n,activation="relu",name=f"spqr_nn_layer_{i}")
            for i,n in enumerate(hidden_layers)
        ]

    def call(self, units):
        for layer in self._layers:
            units = layer(units)
        return units


class BinsLayer(Layer):
    """ BinsLayer class
    """
    def __init__(self,
                 nunits,
                 nbins,
                 border,
                 min_bin_gap=1e-3,
                 name="bins"):
        super().__init__(name=name)
        self._nunits = nunits
        self._nbins = nbins
        self._border = border
        self._min_bin_gap = min_bin_gap
        self._dense = RobustDense(self._nunits * self._nbins, activation=softmax)

    @function
    def call(self, units):
        units = self._dense(units)
        out_shape = concat((shape(units)[:-1], (self._nunits, self._nbins)), 0)
        units = reshape(units, out_shape)
        return units * (2 * self._border - self._nbins * self._min_bin_gap ) - self._min_bin_gap

class SlopesLayer(Layer):
    """ SlopesLayer class
    """
    def __init__(self,
                 nunits,
                 nbins,
                 min_slope=1e-3,
                 name="slopes"):
        super().__init__(name=name)
        self._nunits = nunits
        self._nslopes = nbins - 1
        self._min_slope = min_slope
        self._dense = RobustDense( self._nunits * self._nslopes, activation=softplus)

    @function
    def call(self, units):
        units = self._dense(units)
        out_shape = concat((shape(units)[:-1], (self._nunits, self._nslopes)), 0)
        units = reshape(units, out_shape)
        return units + self._min_slope


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
        self._nn = SplineBlock(hidden_layers=hidden_layers)
        self._min_bin_gap = min_bin_gap
        self._min_slope = min_slope
        self._built = False

    @function
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
        x = self._nn(x)
        return RationalQuadraticSpline(self._bin_widths(x),
                                       self._bin_heights(x),
                                       self._knot_slopes(x),
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

