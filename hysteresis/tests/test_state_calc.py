import pytest
import torch
from hysteresis.base import TorchHysteresis
from hysteresis.linearized import LinearizedHysteresis
from hysteresis.meshing import create_triangle_mesh
from hysteresis.states import get_states, switch, sweep_up, sweep_left, \
    predict_batched_state
import matplotlib.pyplot as plt


class TestStateCalc:
    def test_grad(self):
        h = torch.tensor(1.0, requires_grad=True)
        dummy = h * 4.0
        dummy.backward()

    def test_switches(self):
        mesh = torch.tensor(create_triangle_mesh(1.0))
        h = torch.ones(1, requires_grad=True)
        flip_flag = switch(h, mesh[:, 1])

        flip_flag[0].backward()
        assert h.grad is not None

    def test_sweeps(self):
        mesh = torch.tensor(create_triangle_mesh(0.1))
        h = torch.tensor(0.5, requires_grad=True)
        initial_state = torch.ones_like(mesh[:, 1]) * -1.0
        second_state = sweep_up(h, mesh, initial_state)

        h2 = torch.tensor(0.75, requires_grad=True)
        thrid_state = sweep_up(h2, mesh, second_state)

        h3 = torch.tensor(0.4, requires_grad=True)
        final_state = sweep_left(h3, mesh, thrid_state)

        # final_state[0].backward()
        total = torch.sum(final_state)
        total.backward()
        assert h.grad is not None
        assert h2.grad is not None
        assert h3.grad is not None
        assert not torch.isnan(h.grad)
        assert not torch.isnan(h2.grad)
        assert not torch.isnan(h3.grad)

    def test_get_states(self):
        mesh = torch.tensor(create_triangle_mesh(0.1))
        h = torch.tensor((0.5, 0.75, 0.4, 0.5), requires_grad=True)
        states = get_states(h, mesh)

        total = torch.sum(states[-1])
        total.backward()
        assert not torch.any(torch.isnan(h.grad))

        # plt.tripcolor(*mesh.T, states[-1].detach())
        # plt.show()

    def test_error_handling(self):
        mesh = torch.tensor(create_triangle_mesh(0.1))
        h = torch.tensor((0.5, 0.75, 0.4, 0.5, 1.0, 0.0))
        states = get_states(h, mesh_points=mesh)

        h = torch.tensor((0.5, 1.75, 0.4, 0.5))
        with pytest.raises(RuntimeError):
            states = get_states(h, mesh_points=mesh)

        h = torch.tensor((-0.5, 0.75, 0.4, 0.5))
        with pytest.raises(RuntimeError):
            states = get_states(h, mesh_points=mesh)

    def test_batched(self):
        mesh = torch.tensor(create_triangle_mesh(0.1))
        current_state = torch.ones(mesh.shape[0]) * -1.0
        current_field = -0.01
        h = torch.linspace(0.0, 1.0, 20, requires_grad=True)
        out = predict_batched_state(h.reshape(20, 1, 1),
                                    mesh,
                                    current_state,
                                    current_field)
        out[0][0][0].backward()
        assert not torch.any(torch.isnan(h.grad))


        # idx = 10
        # plt.tripcolor(*mesh.T, out[idx][0].detach())
        # plt.axhline(h[idx].detach())
        # plt.show()
