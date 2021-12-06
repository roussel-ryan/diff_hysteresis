from typing import Callable

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
from pyro.nn import PyroModule, PyroSample
from torch import Tensor


class BayesianHysteresis(PyroModule):
    def __init__(
        self,
        hysteresis_model,
        noise: float = 0.01,
        prior_function: Callable = None,
        kernel_function: Callable = None,
    ) -> None:
        super(BayesianHysteresis, self).__init__()

        self.hysteresis_model = hysteresis_model
        self.noise = noise
        self.prior_function = prior_function
        self.kernel_function = kernel_function

        self.density = PyroSample(self.get_prior_distribution())
        self.scale = PyroSample(dist.Normal(0.5, 0.5))
        self.offset = PyroSample(dist.Normal(0.5, 0.5))
        self.slope = PyroSample(dist.Normal(0.5, 0.5))


    def get_prior_distribution(self) -> dist.Distribution:
        """
        Use prior and kernel functions to specify a distribution for the density vector

        Returns
        -------

        """
        n_mesh_points = self.hysteresis_model.mesh_points.shape[0]

        # use prior function if given
        if self.prior_function is not None:
            prior_mean = self.prior_function(self.hysteresis_model)
            assert (
                prior_mean.shape[0] == n_mesh_points
            ), "prior function used does not match number of mesh points"
        else:
            prior_mean = torch.zeros(n_mesh_points, **self.hysteresis_model.tkwargs)

        # use correlations if function given
        if self.kernel_function is not None:
            covariance_matrix = self.kernel_function(self.hysteresis_model)
            assert (
                covariance_matrix.shape[0] == covariance_matrix.shape[1]
            ), "covariance matrix must be square"
            assert (
                covariance_matrix.shape[0] == n_mesh_points
            ), "covariance matrix size must match mesh points"
        else:
            covariance_matrix = torch.eye(
                n_mesh_points, **self.hysteresis_model.tkwargs
            )

        # return the prior distribution
        return dist.MultivariateNormal(prior_mean, covariance_matrix=covariance_matrix)

    def train(self, **kwargs):
        self.hysteresis_model.train(**kwargs)

    def future(self):
        self.hysteresis_model.future()

    def forward(
            self, X: Tensor,
            Y: Tensor = None
    ) -> Tensor:
        mean = self.hysteresis_model.forward(
            X,
            density_vector=self.density,
            offset=self.offset,
            scale=self.scale,
            slope=self.slope
        )

        # condition on observations
        with pyro.plate("data", len(X)):
            pyro.sample("obs", dist.Normal(mean, self.noise), obs=Y)

        return mean


def positional_covariance(model, sigma):
    mesh_points = model.mesh_points
    x = mesh_points[:, 0]
    y = mesh_points[:, 1]
    n_mesh = len(x)
    # create matrix of distances between points in vector
    n = len(x)
    distances = torch.empty((n, n), **model.tkwargs)

    for ii in range(n):
        for jj in range(n):
            distances[ii][jj] = torch.sqrt((x[ii] - x[jj]) ** 2 + (y[ii] - y[jj]) ** 2)

    # calculate correlation matrix using squared exponential
    corr = torch.exp(-(distances ** 2) / (2.0 * sigma ** 2)) + torch.eye(n_mesh) * 1e-2

    return corr


def low_hysteresis_prior(model):
    # use distance from x=y to get prior mean
    mesh_points = model.mesh_points
    x = mesh_points[:, 0]
    y = mesh_points[:, 1]

    d = torch.abs(x - y) / 2 ** 0.5

    # construct mean
    mean = torch.exp(-d / 0.05)

    # undo softplus transform
    mean = torch.log(torch.exp(mean) - 1.0)
    # plt.pcolor(xx, yy, utils.vector_to_tril(mean, model.n))
    # plt.colorbar()

    return mean
