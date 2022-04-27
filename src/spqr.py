# Marco Riggirello & Antoine Venturini
from tensorflow import expand_dims, reshape, Module
from tensorflow.python.keras.layers import Layer, Dense, Activation
from tensorflow.python.keras.activations import softmax, softplus
from tensorflow_probability.python.bijectors import RealNVP, Chain, RationalQuadraticSpline


class SplineBlock(Layer):
    """ `tf.keras` layers block used for learning parameters
    (knots) of rational quadratic splines.

    Inherits from :class: `tensorflow.keras.layers.Layer`.

    :param nunits: Number of splines.
    :type nunits: int
    :param nbins: Number of bins for each spline. Note that the total number
        of spline parameters is `3*nbins - 1`: `nbins` for x and y bin coordinates
        respectively and `nbins - 1` for slopes.
    :type nbins: int
    :param border: The border of the splines. Spline bins are defined in the interval 
        [-border, border], outside the relation between x and y is `y=x`.
    :type border: float
    :param hidden_layers: Dimensions of each dense layer, defaults to `[512, 512]`.
    :type hidde_layers: list[int], optional
    :param min_bin_gap: Minimum distance between subsequent bins, defaults to `1e-3`.
    :type min_bin_gap: float, optional
    :param min_slope: Mimimum spline slope in each bin, defaults to `1e-3`.
    :type min_slope: float, optional
    """
    def __init__(self,
                 nunits,
                 nbins,
                 border,
                 hidden_layers=[512,512],
                 min_bin_gap=1e-3,
                 min_slope=1e-3):
        """ Constructor method.
        """
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
        """ Returns the units tensor transformed by the neural network.

        :param units: Input tensor.
        :type units: tensorflow.Tensor
        :return: One tensor for bin x coordinates (widths), one for y coordinates
            (heights) and one for slopes.
        :rtype: tensorflow.Tensor
        """
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
    """ Creates a rational quadratic spline with trainable parameters.

    :param nbins: Number of spline bins, defaults to 128.
    :type nbins: int, optional
    :param border: Spline border, defaults to 4.
    :type border: float, optional
    :param hidden_layers: Dimensions of each dense layer, defaults to `[512, 512]`.
    :type hidde_layers: list[int], optional
    :param min_bin_gap: Minimum distance between subsequent bins, defaults to `1e-3`.
    :type min_bin_gap: float, optional
    :param min_slope: Mimimum spline slope in each bin, defaults to `1e-3`.
    :type min_slope: float, optional

    .. note::
        For more informations about rational quadratic spline see
        `the original article <https://arxiv.org/abs/1906.04032>`_ by Durkan et al.
    """
    def __init__(self,
                 nbins=128,
                 border=4,
                 hidden_layers=[512,512],
                 min_bin_gap=1e-3,
                 min_slope=1e-3):
        """ Constructor method.
        """
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
        """ Returns a rational quadratic spline with learnable parameters.

        :param x: The spline input.
        :type x: tensorflow.Tensor
        :param nunits: Number of splines.
        :type nunits: int
        :return: Rational quadratic spline with learnable parameters.
        :rtype: tensorflow_probability.bijectors.RationalQuadraticSpline
        """
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
    """ 
    Neural Spline Flow bijector.

    This is a coupling layer type bijector with rational quadratic
    spline acting as transformer. The coupling layer architecture
    can be defined as a list of masks (or a number of splits)
    that decides which variable is conditioned and which is the conditioner
    in each layer (or better, in each transformation step).
    Suppose we want to transform 3 variables using the mask
    [1,-1], where the negative number indicates the second part
    of the split as conditioner. Two coupling layers will be defined:
    the first one maps the feature :math:`x_1` to itself and acts on features
    :math:`x_0` and :math:`x_2`. A second coupling layer acts on these
    transformed variables :math:`x_0^\prime, x_1, x_2^\prime` masking the
    feature :math:`x_{-1}`, i.e. :math:`x_0` and trnsforming the others.
    The user may want to specify only in how many chunks he wants to split 
    the number of variables.
    For this case the `splits` paramenter is defined: a corresponding 
    number of coupling layers is created, where the *j-th* layer
    has a fration *j / nsplit* of input features masked.

    .. note::
    See `the RealNVP documentation <https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP>`_ for more infos.

    :param masks: list of masks for variables.
    :type masks: list[int]
    :param spline_params: dictionary of parameters for SplineInitializer.
    :type spline_params: dict
    """
    def __init__(self,
                 masks,
                 spline_params = {}
                 ):
        """ Default constructor
        """
        self._spline_params = spline_params
        self._masks = masks
        realnvp_args = [
                dict(num_masked=i, bijector_fn=SplineInitializer(**self._spline_params))
                for i in self._masks
        ]

        self._coupling_layers = [
            RealNVP(**splines, name=f"coupling_layer_{i}")
            for i, splines in enumerate(realnvp_args)
        ]
        super().__init__(bijectors=self._coupling_layers, name="nsf")

    @classmethod
    def from_split_features(cls,
                            nsplits,
                            nfeatures,
                            spline_params={}
                            ):
        """ Constructs a NSF bijector in a simple way.

        :param nsplits: The number of splits.
        :type nsplits: int
        :param nfeatures: The number of features (dimension of the distribution).
        :type nfeatures int:

        :return: A NeuralSplineFlow object

        :raises: ValueError
        """
        if nsplits < 2:
            raise ValueError("You must split your features in at least two pieces.")
        masks = [int(nfeatures/i) for i in range(1-nsplits, nsplits) if i != 0]
        return cls(masks, spline_params=spline_params)
