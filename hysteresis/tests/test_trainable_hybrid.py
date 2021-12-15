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


def load():
    aps_model = torch.load('aps_model.pt')

    train_h = aps_model.history_h.reshape(-1, 1)
    train_m = aps_model.history_m.reshape(-1, 1)

    train_y = torch.sin(train_m * 2 * 3.14 / 40)
    return train_h, train_m, train_y


class TestExactHybridGP:
    def test_init(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten())
        pred_m = H(train_x.flatten(), return_real=True)
        likelihood = GaussianLikelihood()

        model = ExactHybridGP(train_x, train_y, H, likelihood)

        assert torch.all(
            torch.isclose(model.train_inputs[0].squeeze(), train_x.squeeze()))
        assert torch.all(torch.isclose(model.train_targets, train_y))

    def test_train(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten(), polynomial_degree=3)
        for constraint_name, constraint in H.named_constraints():
            print(f'Constraint name: {constraint_name:55} constraint = {constraint}')
        likelihood = GaussianLikelihood()
        model = ExactHybridGP(train_x, train_y.flatten(), H, likelihood)

        mll = ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_model(mll, options={'maxiter': 5})

    def test_predict(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten(), polynomial_degree=3)
        likelihood = GaussianLikelihood()
        model = ExactHybridGP(train_x, train_y.flatten(), H, likelihood)

        # evaluate on training data
        assert isinstance(model(train_x), gpytorch.distributions.MultivariateNormal)

        # evaluate on next data
        test_h = torch.rand(10).reshape(-1, 1, 1).double() + min(train_x)
        model.next()
        post = model(test_h)

        # evaluate on future data
        test_h = torch.rand(10).double() + min(train_x)
        model.future()
        post = model(test_h)

    def test_botorch(self):
        train_x, train_m, train_y = load()
        H = BaseHysteresis(train_x.flatten(), polynomial_degree=3)
        likelihood = GaussianLikelihood()
        model = ExactHybridGP(train_x, train_y.flatten(), H, likelihood)

        test_h = torch.rand(10).reshape(-1, 1, 1).double() + min(train_x)
        acq = UpperConfidenceBound(model, beta=0.1)
        with pytest.raises(HysteresisError):
            acq(test_h)

        model.next()
        acq(test_h)

        # try to optimize
        bounds = torch.tensor([min(train_x), max(train_x)]).reshape(2, 1)
        candidate, _ = optimize_acqf(acq, bounds, 1, 1, 1)
