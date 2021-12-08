from typing import Callable, Dict
from .base import BaseHysteresis
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
from pyro.nn import PyroModule, PyroSample
from torch import Tensor


class BayesianHysteresis(BaseHysteresis, PyroModule):
    def __init__(
            self,
            train_h: Tensor = None,
            train_m: Tensor = None,
            trainable: bool = True,
            tkwargs: Dict = None,
            mesh_scale: float = 1.0,
            mesh_density_function: Callable = None,
            polynomial_degree: int = 1,
            temp: float = 1e-2,
            noise: float = 1e-2
    ):
        super(BayesianHysteresis, self).__init__(
            train_h,
            train_m,
            trainable,
            tkwargs,
            mesh_scale,
            mesh_density_function,
            polynomial_degree,
            temp=temp
        )

        self.noise = noise

        if self.n_mesh_points > 1000:
            raise RuntimeWarning(
                f'More than 1000 mesh points ({self.n_mesh_points}), '
                f'may slow down calculations significantly'
            )

        # add priors to module params
        self._raw_hysterion_density = PyroSample(
            dist.MultivariateNormal(
                torch.zeros(len(self.mesh_points), **self.tkwargs),
                covariance_matrix=torch.eye(len(self.mesh_points), **self.tkwargs)
            )
        )
        self.offset = PyroSample(dist.Normal(0.0, 0.5))
        self.scale = PyroSample(dist.Normal(0.0, 0.5))
        self.slope = PyroSample(dist.Normal(0.0, 0.5))

    def forward(self, X: Tensor, Y: Tensor = None, return_real: bool = False):
        mean = super().forward(X, return_real=return_real)

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
