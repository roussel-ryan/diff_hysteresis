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
