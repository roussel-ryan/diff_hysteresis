from typing import Any, Union

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors import GPyTorchPosterior
from gpytorch.models import GP
from torch import Tensor

from hysteresis.base import HysteresisError
from hysteresis.modes import ModeModule, FITTING, NEXT


class ExactHybridGP(ModeModule, GP):
    num_outputs = 1

    def __init__(self, train_x: Tensor, train_y: Tensor, hysteresis_models, **kwargs):
        super(ExactHybridGP, self).__init__()

        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of samples")

        if len(train_y.shape) != 1:
            raise ValueError(
                "multi output models are not supported, train_y must be a 1D tensor"
            )

        if not isinstance(hysteresis_models, list):
            self.hysteresis_models = torch.nn.ModuleList([hysteresis_models])
        else:
            self.hysteresis_models = torch.nn.ModuleList(hysteresis_models)

        # check if all elements are unique
        if not (len(set(self.hysteresis_models)) == len(self.hysteresis_models)):
            raise ValueError("all hysteresis models must be unique")

        # check that training data is the correct size
        self.input_dim = train_x.shape[-1]
        if self.input_dim != len(self.hysteresis_models):
            raise ValueError("training data must match the number of hysteresis models")
        
        # set hysteresis model history data
        self._set_hysteresis_model_train_data(train_x)

        # set train inputs
        self.train_inputs = (train_x,)

        self.m_transform = Normalize(self.input_dim)

        #train outcome transform
        self.outcome_transform = Standardize(1)
        self.outcome_transform.train()
        self.train_targets = self.outcome_transform(train_y.unsqueeze(1))[0].flatten()
        self.outcome_transform.eval()

        # get magnetization from hysteresis models
        train_m = self.get_magnetization(train_x, mode=FITTING).detach()

        self.gp = SingleTaskGP(
            train_m,
            train_y.unsqueeze(1),
            **kwargs
        )

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def _set_hysteresis_model_train_data(self, train_h):
        for idx, hyst_model in enumerate(self.hysteresis_models):
            hyst_model.set_history(train_h[:, idx])
        
    def apply_fields(self, x: Tensor):
        for idx, hyst_model in enumerate(self.hysteresis_models):
            hyst_model.apply_field(x[:, idx])

    def get_magnetization(self, X, mode=None):
        train_m = []
        # set applied fields and calculate magnetization for training data
        for idx, hyst_model in enumerate(self.hysteresis_models):
            hyst_model.mode = mode or self.mode
            train_m += [hyst_model(X[..., idx], return_real=True)]
        return torch.cat([ele.unsqueeze(-1) for ele in train_m], dim=-1)

    def get_normalized_magnetization(self, X, mode=None):
        m = self.get_magnetization(X, mode)

        # check to see if a normalization model has been trained
        if not self.m_transform.equals(Normalize(self.input_dim)) or self.training:
            return self.m_transform(m)
        else:
            return m

    def posterior(
        self, X: Tensor, observation_noise: Union[bool, Tensor] = False, **kwargs: Any
    ) -> GPyTorchPosterior:
        if self.mode != NEXT:
            raise HysteresisError("calling posterior requires NEXT mode")
        M = self.get_normalized_magnetization(X)

        return self.gp.posterior(M, observation_noise=observation_noise, **kwargs)

    def forward(self, X, from_magnetization=False, return_real=False):
        train_m = self.get_normalized_magnetization(X)

        if self.training:
            self.gp.set_train_data(train_m, self.train_targets)

        if return_real:
            return self.outcome_transform.untransform_posterior(self.gp(
                train_m.unsqueeze(-1)))
        else:
            return self.gp(train_m)
