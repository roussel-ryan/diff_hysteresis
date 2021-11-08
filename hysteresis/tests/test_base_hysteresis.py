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

        #H.predict_magnetization(h_data)

    def test_empty_hysteresis(self):
        H = TorchHysteresis()

    def test_autograd(self):
        h_data = torch.rand(10, requires_grad=True)
        H = TorchHysteresis(h_data)

        h_test = torch.ones(1, requires_grad=True)
        m = H.predict_magnetization(h_new=h_test)
        m[-1].backward()
        assert not torch.isnan(h_test.grad)
