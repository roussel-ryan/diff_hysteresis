from copy import deepcopy

import botorch.posteriors
import gpytorch.distributions
import matplotlib.pyplot as plt
import pytest
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from hysteresis.base import BaseHysteresis
from hysteresis.hybrid import HybridGP


def generate_1D_model():
    train_x = torch.linspace(-10.0, 10.0, 10)
    train_y = train_x**2

    H = BaseHysteresis(train_h=train_x, trainable=False)

    hgp = HybridGP(train_x.reshape(-1, 1), train_y.reshape(-1, 1), H)
    return hgp


def generate_3D_model():
    train_x = torch.cat([torch.linspace(-10.0, 10.0, 10).reshape(-1, 1)] * 3, dim=1)
    train_y = train_x[:, 0] ** 2

    h = BaseHysteresis(train_h=train_x[:, 0], trainable=False)
    H = [deepcopy(h), deepcopy(h), deepcopy(h)]

    hgp = HybridGP(train_x, train_y.reshape(-1, 1), H)
    return hgp


class TestHysteresisGP:
    def test_model_creation(self):
        train_x = torch.cat([torch.linspace(-10.0, 10.0, 10).reshape(-1, 1)] * 3, dim=1)
        train_y = train_x[:, 0] ** 2

        # test 1D
        h = BaseHysteresis(train_h=train_x[:, 0], trainable=False)
        hgp = HybridGP(train_x[:, 0].reshape(-1, 1), train_y.reshape(-1, 1), h)

        # test 3D
        H = [deepcopy(h), deepcopy(h), deepcopy(h)]
        hgp = HybridGP(train_x, train_y.reshape(-1, 1), H)

        # test 3D with identical set of models
        H = [h, h, h]
        with pytest.raises(ValueError):
            hgp = HybridGP(train_x, train_y.reshape(-1, 1), H)

    def test_training(self):
        # generate some dummy data
        hgp = generate_1D_model()
        hgp.train_model()

        # make initial predictions
        hgp.future()
        test_x = torch.rand((10,1))
        with torch.no_grad():
            post = hgp(test_x, untransform_posterior=True)
            assert isinstance(post, botorch.posteriors.TransformedPosterior)

            post2 = hgp(test_x)
            assert isinstance(post2, gpytorch.distributions.Distribution)

        # test 3d
        hgp = generate_3D_model()
        hgp.train_model()

        # make initial predictions
        hgp.future()
        test_x = torch.rand((10, 3))
        with torch.no_grad():
            post = hgp(test_x, untransform_posterior=True)
            assert isinstance(post, botorch.posteriors.TransformedPosterior)

            post2 = hgp(test_x)
            assert isinstance(post2, gpytorch.distributions.Distribution)

    def test_optimization(self):
        # try in 1D
        hgp = generate_1D_model()
        hgp.next()
        UCB = UpperConfidenceBound(hgp, beta=2.0, maximize=False)
        bounds = torch.stack([-1.0 * torch.ones(1), torch.ones(1)])
        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=1,
            raw_samples=1,
        )

        # try with 3D
        hgp = generate_3D_model()
        hgp.next()
        UCB = UpperConfidenceBound(hgp, beta=2.0, maximize=False)
        bounds = torch.stack([-1.0 * torch.ones(3), torch.ones(3)])
        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=1,
            raw_samples=1,
        )

        # try with incorrect mode
        hgp.future()
        UCB = UpperConfidenceBound(hgp, beta=2.0, maximize=False)
        with pytest.raises(ValueError):
            candidate, acq_value = optimize_acqf(
                UCB,
                bounds=bounds,
                q=1,
                num_restarts=1,
                raw_samples=1,
            )