import torch
from hysteresis.base import TorchHysteresis
from hysteresis.linearized import LinearizedHysteresis


class TestBaseHysteresis:
    def test_base_hysteresis(self):
        h_data = torch.rand(10) * 10.0
        H = TorchHysteresis(h_data)

    def test_get_states(self):
        # test recalculation functionality
        h_data = torch.rand(10) * 10.0
        H = TorchHysteresis(h_data)

        _, n_calcs = H.get_states(H.normalize_h(h_data))
        assert n_calcs == 0

        new_h_data = torch.rand(10)

        partial_new_h_data = new_h_data.clone()
        partial_new_h_data[:5] = H.normalize_h(h_data)[:5]
        _, n_calcs = H.get_states(partial_new_h_data)
        assert n_calcs == 5

        partial_new_h_data[:2] = 0.0
        _, n_calcs = H.get_states(partial_new_h_data)
        assert n_calcs == 10

        new_h_data = torch.rand(10)
        _, n_calcs = H.get_states(new_h_data)
        assert n_calcs == 10

    def test_magnetization_prediction(self):
        h_data = torch.rand(10)
        H = TorchHysteresis(h_data)

        h_test = h_data
        H.predict_magnetization(h_test)

    def test_linear_subtraction(self):
        h_data = torch.rand(10)
        m_data = h_data
        H = LinearizedHysteresis(h_data, m_data)

        H.predict_magnetization(h_data)

    def test_empty_hysteresis(self):
        H = TorchHysteresis()

    def test_autograd(self):
        ########## CURRENTLY NOT WORKING ########################
        h_data = torch.rand(10, requires_grad=True)
        H = TorchHysteresis(h_data)

        h_test = torch.ones(1, requires_grad=True) * 0.5
        m = H.predict_magnetization(h_new=h_test)
        m[-1].backward()
