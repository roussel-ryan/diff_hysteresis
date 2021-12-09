from typing import List, Any, Dict

import matplotlib.pyplot as plt
import torch
from botorch.utils.containers import TrainingData
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.model import Model
from hysteresis.modes import ModeEvaluator, NEXT, REGRESSION, FUTURE


class HybridGP(Model, ModeEvaluator):
    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs: Any) -> Dict[
        str, Any]:
        pass

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        pass

    def subset_output(self, idcs: List[int]) -> Model:
        pass

    num_outputs = 1

    def __init__(self, train_x: Tensor, train_y: Tensor, hysteresis_models):
        super(HybridGP, self).__init__()
        self.train_x = train_x
        self.train_y = train_y

        if len(self.train_x.shape) != 2:
            raise ValueError("train_x must be 2D")
        if len(self.train_y.shape) != 2:
            raise ValueError("train_y must be 2D")
        if self.train_x.shape[0] != self.train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of samples")

        if not isinstance(hysteresis_models, list):
            self.hysteresis_models = [hysteresis_models]
        else:
            self.hysteresis_models = hysteresis_models

        # check if all elements are unique
        if not (len(set(self.hysteresis_models)) == len(self.hysteresis_models)):
            raise ValueError('all hysteresis models must be unique')

        self.input_dim = self.train_x.shape[-1]
        if self.input_dim != len(self.hysteresis_models):
            raise ValueError("training data must match the number of hysteresis models")

        self.gp = None
        self.mll = None

        self.train_model()

    def train_model(self):
        self.regression()
        train_m = self.get_magnetization(self.train_x)

        if len(train_m) >= 2:
            bounds = torch.cat(
                (
                    torch.min(train_m, dim=0)[0].unsqueeze(1),
                    torch.max(train_m, dim=0)[0].unsqueeze(1),
                ),
                dim=1,
            ).T

            norm_x = Normalize(self.input_dim, bounds.detach())
            standardize_y = Standardize(1)
        else:
            norm_x = None
            standardize_y = None

        self.gp = SingleTaskGP(
            train_m.detach(),
            self.train_y,
            input_transform=norm_x,
            outcome_transform=standardize_y,
        )
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(self.mll)

    def get_magnetization(self, X):
        train_m = []
        # set applied fields and calculate magnetization for training data
        for idx, hyst_model in enumerate(self.hysteresis_models):
            hyst_model.mode = self.mode
            train_m += [hyst_model(X[..., idx])]
        return torch.cat([ele.unsqueeze(1) for ele in train_m], dim=1)

    def predict_from_magnetization(self, M):
        """Predict output based on magnetization, not applied field H"""
        if M.shape[-1] != self.input_dim:
            raise ValueError(
                "Last dimension of input tensor M must match training data"
            )
        post = self.gp(M.reshape(-1, 1, self.input_dim))
        return self.gp.outcome_transform.untransform_posterior(post)

    def posterior(self, X, **kwargs):
        return self.forward(X)

    def forward(self, X, untransform_posterior=False):
        if X.shape[-1] != len(self.hysteresis_models):
            raise ValueError("test data must match the number of hysteresis models")

        eval_m = self.get_magnetization(X)

        # calculate posterior and un-transform standardization if requested
        post = self.gp(eval_m.reshape(-1, 1, self.input_dim))

        if untransform_posterior:
            return self.gp.outcome_transform.untransform_posterior(post)
        else:
            return post
