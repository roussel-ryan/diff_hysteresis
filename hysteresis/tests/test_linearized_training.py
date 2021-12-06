import matplotlib.pyplot as plt
import torch
from hysteresis.base import TorchHysteresis
from hysteresis.bayesian import BayesianHysteresis
from hysteresis.training import map_bayes


class TestLinearizedTraining:
    def test_linearized(self):
        train_x = torch.linspace(0, 1)
        H = TorchHysteresis(train_x, mesh_scale=1.0)
        train_m = H.predict_magnetization_from_applied_fields().detach()

        B = BayesianHysteresis(H)
        guide, _ = map_bayes(train_x, train_m, B, 1, use_linearized=True)
        plt.show()
