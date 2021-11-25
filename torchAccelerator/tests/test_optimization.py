from copy import deepcopy

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from hysteresis.base import TorchHysteresis
from hysteresis.hybrid import HybridGP
from torchAccelerator.first_order import TorchDrift
from torchAccelerator.hysteresis import HysteresisAccelerator, HysteresisQuad
import matplotlib.pyplot as plt


def objective(R):
    # return (torch.sqrt(R[0,0]) - 2.0e-3)**2 + (torch.sqrt(R[2,2]) - 2.0e-3)**2
    return torch.log(R[0, 0] + R[2, 2])


def density_function(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / 0.005)


class TestOptimization:
    def test_optimization(self):
        # hysteresis model
        H = TorchHysteresis(mesh_scale=0.1, trainable=False)
        dens = density_function(H.mesh_points)
        H.h_min = -1.0
        H.hysterion_density = dens
        H.scale = torch.tensor(5000.0)

        # accelerator model
        # accelerator model
        hmodels = [deepcopy(H), deepcopy(H), deepcopy(H)]

        print(hmodels[0].applied_fields)

        # define quadrupoles
        q1 = HysteresisQuad("q1", torch.tensor(0.01), hmodels[0])
        d1 = TorchDrift("d1", torch.tensor(1.0))
        d2 = TorchDrift("d2", torch.tensor(1.0))
        q2 = HysteresisQuad("q2", torch.tensor(0.01), hmodels[1])
        d3 = TorchDrift("d3", torch.tensor(1.0))
        q3 = HysteresisQuad("q3", torch.tensor(0.01), hmodels[2])

        HA = HysteresisAccelerator([q1, d1, q2, d2, q3, d3])

        initial_beam_matrix = torch.eye(6) * 1.0e-8

        # set x_rms beam size to 1 mm and rms divergence to 0.1 mrad
        initial_beam_matrix[0, 0] = 5.0e-3 ** 2
        initial_beam_matrix[1, 1] = 1.0e-4 ** 2
        initial_beam_matrix[2, 2] = 5.0e-3 ** 2
        initial_beam_matrix[3, 3] = 1.0e-4 ** 2

        # initialize with a couple of points
        train_X = torch.ones((3, 3)) * 0.25
        train_X[0] = train_X[0] * 0.0
        train_X[2] = torch.tensor((0.3, -0.6, 0.3))
        train_Y = torch.empty((3, 1))

        print(train_X)

        for j in range(3):
            HA.apply_fields({'q1': train_X[j, 0],
                             'q2': train_X[j, 1],
                             'q3': train_X[j, 2], })

            # get quad matrices
            print(HA.elements['q2'].get_magnetization_history())
            print(HA.elements['q2'].get_transport_matrix())

            beam_matrix = HA.forward(initial_beam_matrix, full=False)
            train_Y[j] = objective(beam_matrix[-1])

            fig, ax = plt.subplots()
            ax.plot(torch.sqrt(beam_matrix[:, 0, 0]).detach())
            ax.plot(torch.sqrt(beam_matrix[:, 2, 2]).detach())
            ax.set_title(f"{train_X[j]}:{train_Y[j].detach()}")


        plt.show()
