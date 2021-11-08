from torchAccelerator.hysteresis import HysteresisQuad, HysteresisAccelerator
from torchAccelerator.first_order import TorchQuad, TorchDrift, TorchAccelerator
from hysteresis.base import TorchHysteresis
from hysteresis.visualization import plot_hysteresis_density
import torch
import matplotlib.pyplot as plt


def density_function(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / 0.5)


class TestHysteresisQuad:
    def test_hysteresis_quad(self):
        H = TorchHysteresis(mesh_scale=0.1, trainable=False)

        dens = density_function(H.mesh_points)
        H.hysterion_density = dens
        #plot_hysteresis_density(H)

        t = torch.linspace(0, 1, 20)
        test_h = torch.cat((t, t.flipud()))
        K1 = 100.0*0.75*(2*test_h - 1)
        pred = H.predict_magnetization(test_h)

        Q = TorchQuad('q1', torch.tensor(0.01), torch.tensor(0.0))
        HQ = HysteresisQuad('Q1', torch.tensor(0.01), H, scale=100.0)
        D = TorchDrift('d1', torch.tensor(5.0))

        # initial beam matrix
        R = torch.eye(6)

        HA = HysteresisAccelerator([HQ, D])
        A = HysteresisAccelerator([Q, D])
        for name, val in HA.named_parameters():
            print(f'{name}: {val}')
        for name, val in A.named_parameters():
            print(f'{name}: {val}')

        HA.apply_fields({'Q1': test_h[:25]})
        #print(HA.calculate_transport())

        # scan H_fantasy
        out_h = []
        out = []
        for val in test_h.unsqueeze(1):
            HA.Q1.fantasy_H.data = val
            A.q1.K1.data = 100.0*0.75*(2*val - 1)
            R_f = HA(R)
            out_h += [R_f[0, 0].detach()]

            R_f = A(R)
            out += [R_f[0, 0].detach()]

        plt.plot(test_h, out, label='hysteresis_off')
        plt.plot(test_h, out_h, label='hysteresis_on')
        plt.legend()

        plt.show()
