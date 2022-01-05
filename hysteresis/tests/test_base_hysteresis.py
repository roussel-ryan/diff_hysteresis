import pytest
import torch

from hysteresis.base import BaseHysteresis, HysteresisError
from hysteresis.modes import NEXT, FUTURE, REGRESSION, FITTING
from hysteresis.states import get_states


class TestBaseHysteresis:
    def test_empty(self):
        h_data = torch.rand(10)
        m_data = torch.rand(10)

        H = BaseHysteresis()
        with pytest.raises(RuntimeError):
            H(h_data)

        H = BaseHysteresis()
        H.set_history(h_data, m_data)

        # try setting protected variables
        H2 = BaseHysteresis()
        with pytest.raises(AttributeError):
            H2.history_h = h_data
        with pytest.raises(AttributeError):
            H2.history_m = m_data

        H2.future()
        H2(h_data)

        H2.next()
        H2(h_data.reshape(-1, 1, 1))

    def test_constraints(self):
        H = BaseHysteresis()

        with pytest.raises(RuntimeError):
            H.offset = 20000.0

    def test_set_history(self):
        h_data = torch.rand(10) * 10.0
        m_data = torch.rand(10)

        H = BaseHysteresis(h_data, m_data)
        n_grid_pts = len(H.mesh_points)
        assert torch.isclose(
            min(H._history_h), torch.zeros(1).double()
        ) and torch.isclose(max(H._history_h), torch.ones(1).double())
        assert torch.isclose(
            torch.mean(H._history_m), torch.zeros(1).double(), atol=1e-6
        )
        assert torch.isclose(torch.std(H._history_m), torch.ones(1).double(), atol=1e-6)
        assert H._states.shape == torch.Size([10, n_grid_pts])

        # test unnormalized versions
        assert torch.allclose(H.history_h, h_data.double(), atol=1e-5)
        assert torch.allclose(H.history_m, m_data.double(), atol=1e-5)

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
        h_data = torch.linspace(-1.5, 10.0, 20)
        m_data = torch.linspace(-9.5, 10.0, 20)
        H = BaseHysteresis(h_data, m_data)

        # set hysterion density to near zero
        H.hysterion_density = torch.ones(H.n_mesh_points) * 1e-5

        # predict real values with training
        m_pred = H(h_data, return_real=True)
        assert H.mode == FITTING
        assert m_pred.shape == torch.Size([20])
        assert torch.allclose(m_data.double(), m_pred, rtol=1e-2)

        m_pred = H(h_data, return_real=False)
        assert m_pred.shape == torch.Size([20])

    def test_forward_training_w_o_data(self):
        h_data = torch.linspace(-1.0, 10.0)
        H = BaseHysteresis(h_data)

        m_pred = H(h_data, return_real=True)
        assert m_pred.shape == h_data.shape
        
    def test_forward_current(self):
        h_data = torch.linspace(-1.0, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data)
        H.current()
        result = H()
        assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float64), rtol=1e-3)
        
        # should throw error if no data specified
        H2 = BaseHysteresis()
        with pytest.raises(HysteresisError):
            H2.current()
            H2()

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
        assert torch.allclose(
            m_pred_future,
            H.transformer.untransform(
                H.transformer.transform(h_test)[0], m_pred_future_norm
            )[1],
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

        assert torch.allclose(
            m_pred_future,
            H.transformer.untransform(
                H.transformer.transform(h_test)[0], m_pred_future_norm
            )[1],
        )

        bad_inputs = [torch.rand(2, 3, 4), torch.rand(10), torch.rand(10, 1)]
        for ele in bad_inputs:
            with pytest.raises(ValueError):
                res = H(ele)

    def test_applying_fields(self):
        h_data = torch.linspace(1.0, 10.0, 10)
        H = BaseHysteresis(h_data)

        # apply a field to the magnet
        h_new = torch.rand(3) * 5.0 + min(h_data)
        for i in range(len(h_new)):
            h_new_total = torch.cat((h_data, h_new[: i + 1])).double()
            H.apply_field(h_new[i])
            assert torch.allclose(H.history_h, h_new_total)
            assert torch.isclose(H.history_h[-1], h_new[i].double())

            # compare hysterion state shape to ground truth
            states = get_states(
                H.transformer.transform(h_new_total)[0],
                H.mesh_points,
                temp=H.temp,
                tkwargs=H.tkwargs,
            )
            assert torch.allclose(states, H._states, atol=1e-3)

            # test fitting
            H.fitting()
            with pytest.raises(HysteresisError):
                H(h_data)

            # test data
            h_test = torch.rand(10) + min(h_data)

            # test regression
            H.regression()
            res = H(h_test)
            assert res.shape == h_test.shape

            # test future
            H.future()
            res = H(h_test)
            assert res.shape == h_test.shape

            # test next
            H.next()
            res = H(h_test.reshape(-1, 1, 1))
            assert res.shape == h_test.reshape(-1, 1, 1).shape

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
        assert torch.equal(H.valid_domain, torch.tensor((-1, 10.0)))

    def test_save_load(self):
        h_data = torch.linspace(-1, 10.0)
        m_data = torch.linspace(-10.0, 10.0)
        H = BaseHysteresis(h_data, m_data, mesh_scale=0.1)
