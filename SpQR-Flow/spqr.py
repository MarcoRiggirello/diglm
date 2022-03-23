# Marco Riggirello & Antoine Venturini
from tensorflow import function, expand_dims, Module
from tensorflow.python.keras.layers import Layer, Dense, Reshape
from tensorflow.python.keras.activations import softmax, softplus
from tensorflow_probability.python.bijectors import Bijector, RealNVP, Chain, RationalQuadraticSpline

class SplineBlock(Layer):
    """ SplineBlock class
    """
    def __init__(self,
                 hidden_layers=[512,512]):
        super().__init__()
        self._layers = [
            Dense(n,activation="relu",name=f"spqr_nn_layer_{i}")
            for i,n in enumerate(hidden_layers)
        ]

    def call(self, units):
        if units.shape.rank == 1:
            units = expand_dims(units, axis=0)
            reshape_output = lambda x: x[0]
        else:
            reshape_output = lambda x: x
        for layer in self._layers:
            units = layer(units)
        return reshape_output(units)


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
        self._dense = Dense(self._nunits * self._nbins, activation=softmax)
        self._reshape = Reshape((-1, self._nunits * self._nbins))

    def call(self, units):
        if units.shape.rank == 1:
            units = expand_dims(units, axis=0)
            reshape_output = lambda x: x[0]
        else:
            reshape_output = lambda x: x
        units = self._dense(units)
        units = self._reshape(units)
        units = units * (2 * self._border - self._nbins * self._min_bin_gap ) - self._min_bin_gap
        return reshape_output(units)

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
        self._dense = Dense( self._nunits * self._nslopes, activation=softplus)
        self._reshape = Reshape((-1, self._nunits * self._nslopes))

    def call(self, units):
        if units.shape.rank == 1:
            units = expand_dims(units, axis=0)
            reshape_output = lambda x: x[0]
        else:
            reshape_output = lambda x: x
        units = self._dense(units)
        units = self._reshape(units)
        units = units + self._min_slope
        return reshape_output(units)


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
        #self._bin_widths = None
        #self._bin_heights = None
        #self._knot_slopes = None

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

class NeuralSplineFlow(Bijector):
    """ Neural Spline Flow bijector.
    """
    def __init__(self,
                 splits=None,
                 masks=None,
                 **spline_kwargs
                 ):
        super().__init__(forward_min_event_ndims=1, name="nsf")
        if splits is None and masks is None:
            raise ValueError("You must specify 'splits' OR 'masks'.")
        if splits is not None and masks is not None:
            raise ValueError("You can specify `splits` OR `masks`, not both.")
        self._spline_fn = SplineInitializer(**spline_kwargs)
        self._coupling_layers = []
        if splits is not None:
            if splits < 2:
                raise ValueError("nsplits must be greater than or equal to 2 ",
                                 "(You must split you feature vec in at least two parts).")
            for i in range(1,splits):
                self._coupling_layers.append(RealNVP(fraction_masked=i/splits,
                                                         bijector_fn=self._spline_fn))
                self._coupling_layers.append(RealNVP(fraction_masked=-i/splits,
                                                         bijector_fn=self._spline_fn))
        if masks is not None:
            for i in masks:
                self._coupling_layers.append(RealNVP(i, bijector_fn=self._spline_fn))

        self._bijector = Chain(bijectors=self._coupling_layers)

    def _forward(self, x):
        return self._bijector.forward(x)

    def _inverse(self, y):
        return self._bijector.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self._bijector.forward_log_det_jacobian(x)

    def _inverse_log_det_jacobian(self, y):
        return self._bijector.inverse_log_det_jacobian(y)
