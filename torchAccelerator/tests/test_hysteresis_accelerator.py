from copy import deepcopy

import torch

from hysteresis.base import BaseHysteresis
from torchAccelerator.first_order import TorchDrift
from torchAccelerator.hysteresis import HysteresisQuad, HysteresisAccelerator


def density_function(mesh_pts, h = 0.5):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / h)


class TestHysteresisAccelerator:
    def test_basic(self):
        H = BaseHysteresis(
            mesh_scale=0.1,
            trainable=False,
            fixed_domain=torch.tensor((-1.0, 1.0))
        )
        #H.hysterion_density = density_function(H.mesh_points, 0.0001)
        H.slope = 400.0
        H.scale = 0.0
        H.offset = -H.slope / 2.0

        q1 = HysteresisQuad("q1", torch.tensor(0.01), H)
        d1 = TorchDrift("d1", torch.tensor(1.0))

        HA = HysteresisAccelerator([q1, d1])
        init_beam_matrix = torch.eye(6)

        HA.apply_fields({'q1': torch.zeros(1)})

        HA.current()
        m1 = HA(init_beam_matrix)

        HA.apply_fields({'q1': torch.ones(1)})
        m2 = HA(init_beam_matrix)
        assert not torch.equal(m1, m2)

        HA.apply_fields({'q1': torch.zeros(1)})
        m3 = HA(init_beam_matrix)
        assert torch.equal(m1, m3)

        assert torch.equal(
            HA.elements['q1'].hysteresis_model.history_h,
            torch.tensor((0.0, 1.0, 0.0)).double()
        )

    def test_set_histories(self):
        H = BaseHysteresis(
            trainable=False,
            fixed_domain=torch.tensor((-1.0, 1.0))
        )
        # H.hysterion_density = density_function(H.mesh_points, 0.0001)
        H.slope = 400.0
        H.scale = 0.0
        H.offset = -H.slope / 2.0

        q1 = HysteresisQuad("q1", torch.tensor(0.01), deepcopy(H))
        d1 = TorchDrift("d1", torch.tensor(1.0))
        q2 = HysteresisQuad("q2", torch.tensor(0.01), deepcopy(H))
        HA = HysteresisAccelerator([q1, d1, q2])

        new_histories = torch.rand(3, 2).double()
        HA.set_histories(new_histories)
        assert torch.equal(HA.elements['q1'].hysteresis_model.history_h,
                           new_histories[:,0])
        assert torch.equal(HA.elements['q2'].hysteresis_model.history_h,
                           new_histories[:, 1])




