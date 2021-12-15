from typing import Any, Union

import gpytorch.models
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from torch import Tensor

from hysteresis.base import HysteresisError
from hysteresis.modes import ModeEvaluator, NEXT, REGRESSION, FITTING


class ExactHybridGP(ExactGP, GPyTorchModel, ModeEvaluator):
    _num_outputs = 1

    def __init__(self, train_x: Tensor, train_y: Tensor, hysteresis_models, likelihood):
        super(ExactHybridGP, self).__init__(
            torch.empty_like(train_x), torch.empty_like(train_y), likelihood
        )

        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of samples")

        if not isinstance(hysteresis_models, list):
            self.hysteresis_models = torch.nn.ModuleList([hysteresis_models])
        else:
            self.hysteresis_models = torch.nn.ModuleList(hysteresis_models)

        # freeze polynomial fits
        for model in self.hysteresis_models:
            pass
            # model.transformer.freeze()

        # check if all elements are unique
        if not (len(set(self.hysteresis_models)) == len(self.hysteresis_models)):
            raise ValueError("all hysteresis models must be unique")

        self.input_dim = train_x.shape[-1]
        if self.input_dim != len(self.hysteresis_models):
            raise ValueError("training data must match the number of hysteresis models")

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # set training data
        self.set_train_data(train_x, train_y, strict=False)

    def apply_fields(self, x: Tensor):
        for idx, hyst_model in enumerate(self.hysteresis_models):
            hyst_model.apply_field(x[:, idx])

    def get_magnetization(self, X, mode=None):
        train_m = []
        # set applied fields and calculate magnetization for training data
        for idx, hyst_model in enumerate(self.hysteresis_models):
            hyst_model.mode = mode or self.mode
            train_m += [hyst_model(X[..., idx], return_real=True)]
        return torch.cat([ele.unsqueeze(1) for ele in train_m], dim=1)

    def predict_from_magnetization(self, m):
        train_inputs_m = self.get_magnetization(self.train_inputs[0]).reshape(
            -1, 1, len(self.hysteresis_models)
        )
        total_m = torch.vstack(
            (train_inputs_m, m.reshape(-1, 1, len(self.hysteresis_models)))
        )
        mean_m = self.mean_module(total_m)
        covar_m = self.covar_module(total_m)

        return MultivariateNormal(mean_m, covar_m)

    def posterior(
        self, X: Tensor, observation_noise: Union[bool, Tensor] = False, **kwargs: Any
    ) -> GPyTorchPosterior:
        if self.mode != NEXT:
            raise HysteresisError("calling posterior requires NEXT mode")
        return super(ExactHybridGP, self).posterior(X, observation_noise, **kwargs)

    def fitting(self):
        super(ExactHybridGP, self).fitting()
        self.train()

    def next(self):
        super(ExactHybridGP, self).next()
        self.eval()

    def future(self):
        super(ExactHybridGP, self).future()
        self.eval()

    def regression(self):
        super(ExactHybridGP, self).regression()
        self.eval()

    def forward(self, X, from_magnetization=False):
        if self.mode != FITTING:
            self.eval()
        else:
            self.train()

        if X.shape[-1] != len(self.hysteresis_models):
            raise ValueError("test data must match the number of hysteresis models")

        # get magnetization from training data (applied fields) if training the
        # applied fields must exactly match the history (mode==FITTING)
        if self.training:
            train_m = self.get_magnetization(self.train_inputs[0], mode=FITTING)
        else:
            # if X is not the same shape as train_inputs then we need to get the
            # subvector - else just use training data
            if self.train_inputs[0].shape != X.shape:
                # if we are using NEXT mode we need to get a single batch sample
                if self.mode == NEXT:
                    train_x = X[0][: len(self.train_inputs[0])]
                else:
                    train_x = X[: len(self.train_inputs[0])]
            else:
                train_x = self.train_inputs[0]

            train_m = self.get_magnetization(train_x, mode=REGRESSION)

        # if we are calculating from magnetization, append the magnetization samples
        if from_magnetization:
            total_m = torch.vstack(
                (train_m, X[len(self.train_inputs[0]) :].reshape(-1, 1))
            )
        else:
            if self.training or self.train_inputs[0].shape == X.shape:
                total_m = train_m
            else:
                assert self.mode != FITTING
                if self.mode == NEXT:
                    eval_x = X[:, len(self.train_inputs[0]) :, :].unsqueeze(-2)
                else:
                    eval_x = X[len(self.train_inputs[0]) :]

                eval_m = self.get_magnetization(eval_x, mode=self.mode)
                total_m = torch.vstack((train_m, eval_m.reshape(-1, 1)))

        mean_m = self.mean_module(total_m)
        covar_m = self.covar_module(total_m)

        return MultivariateNormal(mean_m, covar_m)
