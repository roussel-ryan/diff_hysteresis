import matplotlib.pyplot as plt
import numpy as np
import torch
import utils

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import hysteresis


class BayesianHysteresis(PyroModule):
    def __init__(self, h_data, h_min, h_max, b_sat, n):
        super(BayesianHysteresis, self).__init__()

        self.hysteresis_model = hysteresis.Hysteresis(h_data,
                                                      h_min,
                                                      h_max,
                                                      b_sat,
                                                      n,
                                                      trainable=False)

        # vector length
        vector_length = utils.get_upper_trainagle_size(n)

        # represent the hysterion density inside [0,1] with a beta distribution
        self.density = PyroSample(dist.Normal(0, 1).expand([vector_length]).to_event(1))

        # represent the scale and offset with Normal distributions - priors assume
        # normalized output
        self.scale = PyroSample(dist.Normal(1.0, 1.0))
        self.offset = PyroSample(dist.Normal(0.0, 1.0))

    def forward(self, x, y=None):

        # set hysteresis model parameters to do calculation
        self.hysteresis_model._raw_hyst_density_vector = self.density.double().flatten()

        self.hysteresis_model.scale = torch.nn.Softplus()(self.scale)
        self.hysteresis_model.offset = self.offset

        # do prediction
        mean = self.hysteresis_model.predict_magnetization(h=x)

        # condition on observations
        with pyro.plate('data', x.shape[0]):
            obs = pyro.sample('obs', dist.Normal(mean, 0.1), obs=y)
        return mean
