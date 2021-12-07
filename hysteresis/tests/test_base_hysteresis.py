import matplotlib.pyplot as plt
import pytest
import torch
from hysteresis.base import BaseHysteresis


class TestBaseHysteresis:
    def test_set_history(self):
        h_data = torch.rand(10) * 10.0
        m_data = torch.rand(10)

        H = BaseHysteresis(h_data, m_data)
        n_grid_pts = len(H.mesh_points)
        assert torch.min(H._history_h) == 0.0 and torch.max(H._history_h) == 1.0
        assert torch.isclose(torch.mean(H._history_m), torch.zeros(1), atol=1e-6)
        assert torch.isclose(torch.std(H._history_m), torch.ones(1), atol=1e-6)
        assert H._states.shape == torch.Size([10, n_grid_pts])

    def test_negative_saturation(self):
        h_data = torch.linspace(0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)
        assert torch.isclose(H.get_negative_saturation(), torch.min(m_data), rtol=1e-2)

    def test_forward_training(self):
        h_data = torch.linspace(0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)

        # predict with training
        m_pred = H(h_data)
        assert m_pred.shape == torch.Size([100])

    def test_forward_future(self):
        h_data = torch.linspace(0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)

        # predict future
        h_test = h_data.flipud()
        H.future()
        assert H._mode == 'future'
        m_pred_future_norm = H(h_test, return_norm=True)
        m_pred_future = H(h_test)
        assert m_pred_future_norm.shape == torch.Size([100])
        assert torch.all(
            torch.isclose(
                m_pred_future,
                H.transformer.untransform(
                    H.transformer.transform(h_test)[0],
                    m_pred_future_norm
                )[1]
            )
        )

    def test_forward_next(self):
        h_data = torch.linspace(0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)

        # predict future
        h_test = h_data.flipud().reshape(-1, 1, 1)
        H.next()
        assert H._mode == 'next'
        m_pred_future_norm = H(h_test, return_norm=True)
        m_pred_future = H(h_test)
        assert m_pred_future_norm.shape == torch.Size([100, 1, 1])
        assert torch.all(
            torch.isclose(
                m_pred_future,
                H.transformer.untransform(
                    H.transformer.transform(h_test)[0],
                    m_pred_future_norm
                )[1]
            )
        )

        with pytest.raises(ValueError):
            res = H(torch.rand(2, 3, 4))

    def test_autograd(self):
        h_data = torch.linspace(0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)
        H.next()

        with torch.autograd.detect_anomaly():
            h_test = torch.tensor(0.0, requires_grad=True)
            m = H(h_test.reshape(1, 1, 1))
            m.backward()

        assert not torch.isnan(h_test.grad)
