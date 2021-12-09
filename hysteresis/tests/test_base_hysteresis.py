import matplotlib.pyplot as plt
import pytest
import torch
from hysteresis.base import BaseHysteresis
from hysteresis.visualization import plot_hysteresis_density
from hysteresis.modes import NEXT, FUTURE, REGRESSION


class TestBaseHysteresis:
    def test_empty(self):
        h_data = torch.rand(10) * 10.0
        m_data = torch.rand(10)

        H = BaseHysteresis()
        H.set_history(h_data, m_data)

        H2 = BaseHysteresis()
        with pytest.raises(AttributeError):
            H.history_h = h_data
        with pytest.raises(AttributeError):
            H.history_m = m_data

    def test_set_history(self):
        h_data = torch.rand(10) * 10.0
        m_data = torch.rand(10)

        H = BaseHysteresis(h_data, m_data)
        n_grid_pts = len(H.mesh_points)
        assert torch.min(H._history_h) == 0.0 and torch.max(H._history_h) == 1.0
        assert torch.isclose(
            torch.mean(H._history_m), torch.zeros(1).double(), atol=1e-6
        )
        assert torch.isclose(
            torch.std(H._history_m), torch.ones(1).double(), atol=1e-6
        )
        assert H._states.shape == torch.Size([10, n_grid_pts])

        # test unnormalized versions
        assert torch.all(torch.isclose(H.history_h, h_data.double(), atol=1e-5))
        assert torch.all(torch.isclose(H.history_m, m_data.double(), atol=1e-5))

        # test for gradients
        for ele in [H._history_h, H._history_m, H.history_h, H.history_m]:
            assert ele.grad is None

    def test_equal(self):
        h_data = torch.rand(10)
        m_data = h_data

        with pytest.raises(RuntimeError):
            H = BaseHysteresis(h_data, m_data)

    def test_negative_saturation(self):
        h_data = torch.linspace(0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)
        assert torch.isclose(
            H.get_negative_saturation(), torch.min(m_data).double(), rtol=1e-2
        )

    def test_saturations(self):
        h_data = torch.linspace(0, 10.0)
        H = BaseHysteresis(h_data, mesh_scale=0.1)
        m_pred = H(h_data)

    def test_forward_training(self):
        h_data = torch.linspace(-1.0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)

        # set hysterion density to near zero
        H.hysterion_density = torch.ones(H.n_mesh_points) * 1e-5

        # predict real values with training
        m_pred = H(h_data, return_real=True)
        assert H.mode == REGRESSION
        assert m_pred.shape == torch.Size([100])
        assert torch.all(torch.isclose(m_data.double(), m_pred, atol=1e-3))

        m_pred = H(h_data, return_real=False)
        assert m_pred.shape == torch.Size([100])

    def test_forward_training_w_o_data(self):
        h_data = torch.linspace(-1.0, 10.0)
        H = BaseHysteresis(h_data)

        m_pred = H(h_data, return_real=True)

    def test_forward_future(self):
        h_data = torch.linspace(-1.0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)

        # predict future
        h_test = h_data.flipud()
        H.future()
        assert H.mode == FUTURE
        m_pred_future_norm = H(h_test)
        m_pred_future = H(h_test, return_real=True)
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
        h_data = torch.linspace(-1, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)

        # predict future
        h_test = h_data.flipud().reshape(-1, 1, 1)
        H.next()
        assert H.mode == NEXT
        m_pred_future_norm = H(h_test)
        m_pred_future = H(h_test, return_real=True)
        assert m_pred_future_norm.shape == torch.Size([100, 1, 1])
        assert m_pred_future.shape == torch.Size([100, 1, 1])

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
        h_data = torch.linspace(-1, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)
        H.next()

        with torch.autograd.detect_anomaly():
            h_test = torch.tensor(0.0, requires_grad=True)
            m = H(h_test.reshape(1, 1, 1))
            m.backward()

        assert not torch.isnan(h_test.grad)

    def test_valid_domain(self):
        h_data = torch.linspace(-1, 10.0)
        H = BaseHysteresis(train_h=h_data)
        assert torch.equal(H.valid_domain, torch.tensor((-1, 10.0)).double())
