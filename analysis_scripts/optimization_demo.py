import matplotlib.pyplot as plt

from hysteresis.base import TorchHysteresis
from torchAccelerator.hysteresis import HysteresisAccelerator, HysteresisQuad
from torchAccelerator.first_order import TorchDrift
import torch
from copy import deepcopy


def density_function(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / 0.5)


def optimize(accelerator_model, initial_beam_matrix):
    optimizer = torch.optim.Adam(accelerator_model.parameters(), lr=0.1)

    iterations = 2
    for i in range(iterations):
        # zero gradients
        optimizer.zero_grad()

        # calculate loss function from model - R11 of final beam matrix
        loss = accelerator_model.q1.forward()[0, 0]
        print(loss)

        # calculate gradient
        loss.backward()
        print(accelerator_model.q1.fantasy_H.grad)

        # take step with optimizer to determine next point
        optimizer.step()


#hysteresis model
H = TorchHysteresis(mesh_scale=0.1, trainable=False)
dens = density_function(H.mesh_points)
H.h_min = -1.0
H.hysterion_density = dens

# define quadrupoles
q1 = HysteresisQuad('q1', torch.tensor(0.1), deepcopy(H), scale=torch.tensor(100.0))
d1 = TorchDrift('d1', torch.tensor(1.0))


HA = HysteresisAccelerator([q1, d1])

for name, val in HA.named_parameters():
    print(f'{name}:{val}')

R = torch.eye(6)
optimize(HA, R)
