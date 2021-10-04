import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
from pyro.nn import PyroModule, PyroSample

import utils


def calculate_distances(model):
    x = utils.tril_to_vector(model._xx, model._n)
    y = utils.tril_to_vector(model._yy, model._n)

    #create matrix of distances between points in vector
    n = len(x)
    distances = torch.empty((n, n))

    for ii in range(n):
        for jj in range(n):
            distances[ii][jj] = torch.sqrt((x[ii] - x[jj])**2 + (y[ii] - y[jj])**2)

    return distances


class CorrelatedBayesianHysteresis(PyroModule):
    def __init__(self, model, n, sigma):
        super(CorrelatedBayesianHysteresis, self).__init__()

        self.hysteresis_model = model
        # vector length
        vector_length = utils.get_upper_trainagle_size(n)

        # calculate distances between locations
        distances = calculate_distances(self.hysteresis_model)

        # calculate correlation matrix using squared exponential
        corr = torch.exp(-distances**2 / (2.0*sigma**2)) \
               + torch.eye(vector_length)*1e-2

        # represent the hysterion density as a correlated multivariate gaussian
        self.density = PyroSample(
            dist.MultivariateNormal(torch.ones(vector_length),
                                    covariance_matrix=10.0*corr))

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
