import pyro
import torch
from pyro import poutine

from pyro.infer.autoguide import AutoDelta
from hysteresis.bayesian_utils import predict
from hysteresis.bayesian import BayesianHysteresis
from hysteresis.training import train_bayes, map_bayes


class TestBayesianHysteresis:
    def test_init(self):
        h_data = torch.rand(10) * 10.0
        m_data = h_data ** 2

        B = BayesianHysteresis(h_data, m_data)
        m_pred = B(h_data)

    def test_sample_train(self):
        h_data = torch.rand(10) * 10.0
        m_data = h_data

        B = BayesianHysteresis(h_data, m_data)

        results, samples = predict(h_data, B, AutoDelta(B), num_samples=10)
        assert list(results.keys()) == [
            'scale',
            'offset',
            'slope',
            'obs',
            '_RETURN',
            'density'
        ]
        assert torch.all(samples['density'] > 0)
        assert samples['density'].shape == torch.Size([10, 1, len(B.mesh_points)])

        # test training with MAP
        map_bayes(h_data, m_data, B, 100)

    def test_save_load(self):
        pass


