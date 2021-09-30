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
    def __init__(self, model, n):
        super(BayesianHysteresis, self).__init__()

        self.hysteresis_model = model
        # vector length
        vector_length = utils.get_upper_trainagle_size(n)

        # represent the hysterion density inside [0,1] with a beta distribution
        self.density = PyroSample(dist.Normal(0.0, 10.).expand([
            vector_length]).to_event(1))

        # represent the scale and offset with Normal distributions - priors assume
        # normalized output
        self.scale = PyroSample(dist.Normal(1.0, 1.0))
        self.offset = PyroSample(dist.Normal(0.0, 1.0))

    def forward(self, x, y=None):
        # set hysteresis model parameters to do calculation
        raw_vector = self.density.double().flatten()
        scale = torch.nn.Softplus()(self.scale)

        # do prediction
        mean = \
            self.hysteresis_model.predict_magnetization(h=x,
                                                        raw_dens_vector=raw_vector,
                                                        scale=scale,
                                                        offset=self.offset)

        # condition on observations
        with pyro.plate('data', x.shape[0]):
            obs = pyro.sample('obs', dist.Normal(mean, 0.01), obs=y)
        return mean
