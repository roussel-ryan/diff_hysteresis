import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from hysteresis.base import TorchHysteresis
from hysteresis.bayesian import BayesianHysteresis
from hysteresis.training import train_bayes


class TestBayesianHysteresis:
    def test_bayesian_hysteresis(self):
        h_data = torch.rand(10) * 10.0
        m_data = h_data ** 2

        H = TorchHysteresis(h_data)
        B = BayesianHysteresis(H)

        result = B.forward(h_data)
        result = B.forward(h_data, m_data)

    def test_bayesian_training(self):
        h_data = torch.rand(10) * 10.0
        m_data = h_data ** 2

        H = TorchHysteresis(h_data)
        B = BayesianHysteresis(H)

        guide, loss = train_bayes(h_data, m_data, B, 10)


