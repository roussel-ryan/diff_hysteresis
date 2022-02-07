from copy import deepcopy

import gpytorch.distributions
import pytest
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from hysteresis.base import BaseHysteresis, HysteresisError
from hysteresis.hybrid import ExactHybridGP
import hysteresis
import os


def load():
    aps_model = torch.load(
        os.path.join(hysteresis.__file__.split("__init__")[0], "tests", "aps_model.pt")
    )

    train_h = aps_model.history_h.reshape(-1, 1)
    train_m = aps_model.history_m.reshape(-1, 1)

    train_y = torch.sin(train_m * 2 * 3.14 / 40)
    return train_h, train_m, train_y


class TestExactHybridGP:
    def test_init(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten())
        pred_m = H(train_x.flatten(), return_real=True)

        model = ExactHybridGP(train_x, train_y.flatten(), H)

    def test_eval(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten(), polynomial_degree=3)

        model = ExactHybridGP(train_x, train_y.flatten(), H)
        with torch.no_grad():
            post = model(train_x)

    def test_train_basic(self):
        train_x, train_m, train_y = load()

        # try with and without trainable H model
        for ele in [True, False]:
            H = BaseHysteresis(train_x.flatten(), polynomial_degree=1, trainable=ele)

            model = ExactHybridGP(train_x, train_y.flatten(), H)
            mll = ExactMarginalLogLikelihood(model.gp.likelihood, model)

            # requirement for training.py
            value = mll(model(model.train_inputs[0]), model.train_targets)
            assert value.shape == torch.Size([])

            fit_gpytorch_model(mll, options={"maxiter": 2})

        # train with random data inside bounds
        H = BaseHysteresis(train_x.flatten(), polynomial_degree=1)

        model = ExactHybridGP(
            torch.rand(10).unsqueeze(1).double() + torch.min(train_x),
            torch.rand(10).double(),
            H,
        )

        mll = ExactMarginalLogLikelihood(model.gp.likelihood, model)
        fit_gpytorch_model(mll, options={"maxiter": 2})

        # train multi dim with random data
        model = ExactHybridGP(
            torch.rand(10, 3).double() + torch.min(train_x),
            torch.rand(10).double(),
            [deepcopy(H), deepcopy(H), deepcopy(H)],
        )

        mll = ExactMarginalLogLikelihood(model.gp.likelihood, model)
        fit_gpytorch_model(mll, options={"maxiter": 5})

    def test_predict(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten(), polynomial_degree=3)
        model = ExactHybridGP(train_x, train_y.flatten(), H)

        # evaluate on training.py data
        assert isinstance(model(train_x), gpytorch.distributions.MultivariateNormal)

        for ele in [True, False]:
            # evaluate on next data
            test_h = torch.rand(10).unsqueeze(-1).double() + min(train_x)
            model.next()
            post = model(test_h, return_real=ele)

            # evaluate on future data
            test_h = torch.rand(10).double() + min(train_x)
            model.future()
            post = model(test_h.unsqueeze(1), return_real=ele)

    def test_botorch(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten(), polynomial_degree=1)
        model = ExactHybridGP(train_x, train_y.flatten(), H)

        test_h = torch.rand(10).reshape(-1, 1, 1).double() + min(train_x)
        acq = UpperConfidenceBound(model, beta=0.1)
        with pytest.raises(HysteresisError):
            acq(test_h)

        model.next()

        # try to optimize
        bounds = torch.tensor([min(train_x), max(train_x)]).reshape(2, 1)
        candidate, _ = optimize_acqf(acq, bounds, 1, 1, 2)

    def test_multiple_magents(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten())

        with pytest.raises(ValueError):
            h_list = [H, H, H]
            model = ExactHybridGP(train_x, train_y.flatten(), h_list)

        h_list = [deepcopy(H), deepcopy(H), deepcopy(H)]
        with pytest.raises(ValueError):
            model = ExactHybridGP(train_x, train_y, h_list)

        train_x = train_x.expand(61, 3)
        model = ExactHybridGP(train_x, train_y.flatten(), h_list)

        # test fitting eval
        result = model(train_x)
        assert isinstance(result, gpytorch.distributions.MultivariateNormal)

        # test regression eval
        model.regression()
        test_x = torch.rand(6, 3).double() + min(train_x[0])
        result = model(test_x)

        # test forward eval
        model.future()
        result = model(test_x)

        # test next eval
        model.next()
        result = model(test_x.reshape(6, 1, 3))
        result = model(torch.rand(3, 4, 1, 3))

        acq = UpperConfidenceBound(model, beta=0.1)
        acq(test_x.unsqueeze(-2))

        # try to optimize
        bounds = torch.tensor(
            [
                [min(train_x[0]), max(train_x[0])],
                [min(train_x[0]), max(train_x[0])],
                [min(train_x[0]), max(train_x[0])],
            ]
        ).reshape(2, 3)
        candidate, _ = optimize_acqf(acq, bounds, 1, 1, 1)
