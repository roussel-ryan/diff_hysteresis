import matplotlib.pyplot as plt
import torch
from hysteresis.base import TorchHysteresis
from hysteresis.linearized import LinearizedHysteresis


class TestBaseHysteresis:
    def test_base_hysteresis(self):
        h_data = torch.rand(10) * 10.0
        H = TorchHysteresis(h_data)

    def test_magnetization_prediction(self):
        h_data = torch.rand(10)
        H = TorchHysteresis(h_data)

        h_test = h_data
        H.predict_magnetization(h_test)

    def test_linear_subtraction(self):
        h_data = torch.rand(10)
        m_data = h_data
        H = LinearizedHysteresis(h_data, m_data)

        # H.predict_magnetization(h_data)

    def test_empty_hysteresis(self):
        H = TorchHysteresis()

    def test_autograd(self):
        h_data = torch.linspace(0, 1.0, 10)
        H = TorchHysteresis(h_data, mesh_scale=0.1, temp=1e-3)

        with torch.autograd.detect_anomaly():
            h_test = torch.tensor(0.0, requires_grad=True)
            m = H.predict_magnetization(h=h_test)
            m[-1].backward()

        h_test = torch.tensor(0.5, requires_grad=True)
        m = H.predict_magnetization(h_new=h_test)
        m[-1].backward()

        dx = torch.tensor(0.001)
        m_dx = H.predict_magnetization(h_new=h_test + dx)
        num_grad = (m_dx[-1] - m[-1]) / dx
        print(h_test.grad)
        print(num_grad)

        assert not torch.isnan(h_test.grad)

    def test_future_prediction(self):
        h_data = torch.linspace(0.0, 0.5, 10)
        H = TorchHysteresis(mesh_scale=0.1)
        H.applied_fields = h_data

        h_test = torch.linspace(0.6, 1.0, 5)
        m_future = H.predict_magnetization_future(h_test)

        m_correct = torch.tensor([-0.0212, 0.2922, 0.6162, 0.9528, 1.2987]).double()
        assert torch.all(torch.isclose(m_future.detach(), m_correct, rtol=0.01))
