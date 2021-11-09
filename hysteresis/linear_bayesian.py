from typing import Callable

import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample
from torch import Tensor

from .bayesian import BayesianHysteresis


class BayesianLinearizedHysteresis(BayesianHysteresis):
    def __init__(
        self,
        hysteresis_model,
        noise: float = 0.01,
        prior_function: Callable = None,
        kernel_function: Callable = None,
    ) -> None:
        super(BayesianLinearizedHysteresis, self).__init__(
            hysteresis_model, noise, prior_function, kernel_function
        )
        del self.scale
        del self.offset
        pyro.clear_param_store()

        self.l_scale = PyroSample(dist.Normal(2.0, 0.5))
        self.l_slope = PyroSample(dist.Normal(8.0, 0.5))
        self.l_offset = PyroSample(dist.Normal(4.0, 0.5))

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        mean = self.hysteresis_model.predict_magnetization(
            h=X,
            density_vector=self.density,
            l_slope=self.l_slope,
            l_offset=self.l_offset,
            l_scale=self.l_scale,
            real_m=False,
        )

        # condition on observations
        with pyro.plate("data", len(X)):
            pyro.sample("obs", dist.Normal(mean, self.noise), obs=Y)

        return mean
