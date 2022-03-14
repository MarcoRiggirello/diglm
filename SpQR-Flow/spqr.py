# Marco Riggirello & Antoine Venturini

from tensorflow_probability.python.bijectors import Bijector, RealNVP, Chain

from utils import SplineInitializer

class NeuralSplineFlow(Bijector):
    def __init__(self,
                 splits=None,
                 masks=None,
                 spline_fn=SplineInitializer(),
                 name="neural_spline_flow"
                 ):
        super().__init__(forward_min_event_ndims=0,
                         name=name)
        if splits is not None and masks is not None:
            raise ValueError("You can specify `splits` OR `masks`, not both.")
        if splits is None and masks is None:
            raise ValueError("You must specify 'splits' OR 'masks'.")
        self._spline_fn = spline_fn
        self._coupling_layers = []
        if splits is not None:
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
