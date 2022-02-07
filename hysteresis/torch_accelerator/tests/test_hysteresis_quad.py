from hysteresis.torch_accelerator.hysteresis import HysteresisQuad
from hysteresis.torch_accelerator.first_order import TorchQuad
from hysteresis.torch_accelerator.hysteresis import BaseHysteresis
import torch


def density_function(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / 0.5)


class TestHysteresisQuad:
    def test_normal_quad_grad(self):
        Q = TorchQuad("q1", torch.tensor(1.0), torch.tensor(1.0))
        M = Q.forward()
        M[0, 0].backward()
        assert Q.K1.grad == -torch.sin(torch.tensor(1.0)) / 2.0

        new_tensor = torch.tensor(-0.5, requires_grad=True)
        M = Q.get_matrix(new_tensor)
        M[1, 0].backward()
        assert torch.isclose(new_tensor.grad, torch.tensor(-1.173), atol=1e-4)

        new_tensor = torch.tensor(-100.5, requires_grad=True)
        M = Q.get_matrix(new_tensor)
        M[0, 0].backward()

    def test_hysteresis_quad(self):
        H = BaseHysteresis(mesh_scale=0.1)
        HQ = HysteresisQuad("Q1", torch.tensor(1.0), H, scale=torch.tensor(1.0))

        # test in next mode
        HQ.next()
        result = HQ(torch.rand(3, 1, 1))

    def test_hysteresis_quad_grad(self):
        with torch.autograd.detect_anomaly():
            h_data = torch.linspace(0, 1.0, 10)
            H = BaseHysteresis(h_data, mesh_scale=0.1)
            HQ = HysteresisQuad("Q1", torch.tensor(1.0), H, scale=torch.tensor(1.0))

            # test gradient for calculating the transport matrix from magnetization
            x = torch.tensor(0.5, requires_grad=True)
            matrix = HQ._calculate_beam_matrix(x)
            matrix[0, 0].backward()
            assert not torch.isnan(x.grad)

            x = torch.tensor(0.0, requires_grad=True)
            matrix = HQ._calculate_beam_matrix(x)
            matrix[0, 0].backward()
            assert not torch.isnan(x.grad)

            x = torch.tensor(-0.5, requires_grad=True)
            matrix = HQ._calculate_beam_matrix(x)
            matrix[0, 0].backward()
            assert not torch.isnan(x.grad)

            # test calculating magnetization
            HQ.future()
            x = torch.tensor(0.0, requires_grad=True)
            m = HQ.hysteresis_model(torch.atleast_1d(x))
            m[0].backward()
            assert not torch.isnan(x.grad)

            # test gradient for calculating the transport matrix from applied field
            x = torch.tensor(0.2, requires_grad=True)
            matrix = HQ.get_transport_matrix(torch.atleast_1d(x))
            matrix[0, 0, 0].backward()
            assert not torch.isnan(x.grad)

    def test_autograd_w_applied_fields(self):
        with torch.autograd.detect_anomaly():
            h_data = torch.linspace(0, 1.0, 10)
            H = BaseHysteresis(h_data, mesh_scale=0.1)
            HQ = HysteresisQuad("Q1", torch.tensor(1.0), H, scale=torch.tensor(1.0))

            # apply new field
            new_H = torch.tensor(0.5)
            HQ.apply_field(new_H)

            # predict with current
            HQ.current()
            assert HQ().shape == torch.Size([6, 6])
