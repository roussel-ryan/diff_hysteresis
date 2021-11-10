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


def beamsize(accel_model, R_i):
    return accel_model.forward(R_i)[0, 0]


def optimize(accelerator_model, initial_beam_matrix, apply_fields=False):
    optimizer = torch.optim.SGD(accelerator_model.parameters(), lr=0.005)

    iterations = 50
    points = []
    losses = []
    for i in range(iterations):
        # zero gradients
        optimizer.zero_grad()

        # calculate loss function from model - R11 of final beam matrix
        loss = beamsize(accelerator_model, initial_beam_matrix)
        losses += [loss.detach()]
        points += [accelerator_model.q1.fantasy_H.clone().detach()]
        print(points[-1])

        # calculate gradient
        with torch.autograd.detect_anomaly():
            loss.backward()

        # take step with optimizer to determine next point
        optimizer.step()

        if apply_fields:
            accelerator_model.apply_fields({'q1': points[-1]})

    return points, losses


# hysteresis model
H = TorchHysteresis(mesh_scale=0.1, trainable=False)
dens = density_function(H.mesh_points)
H.h_min = -1.0
H.hysterion_density = dens

# define quadrupoles
q1 = HysteresisQuad("q1", torch.tensor(0.1), deepcopy(H), scale=torch.tensor(100.0))
d1 = TorchDrift("d1", torch.tensor(1.0))

HA = HysteresisAccelerator([q1, d1])
HA.q1.fantasy_H.data = torch.tensor(-0.5)

for name, val in HA.named_parameters():
    if val.requires_grad:
        print(f'{name}:{val}')

# do optimization
R = torch.eye(6)
p, l = optimize(HA, R, True)
plt.plot(p, l, 'o-')

# plot beam size as a function of fantasy_H
h_f = torch.linspace(-1.0, 1, 100)
bs = []
bs_grad = []
for ele in h_f:
    HA.q1.fantasy_H.data = ele
    bs += [beamsize(HA, R).detach()]

    HA(R)[0, 0].backward()
    bs_grad += [HA.q1.fantasy_H.grad.clone()]
    HA.q1.fantasy_H.grad.zero_()

plt.plot(h_f, bs, label="step 0")
plt.figure()
plt.plot(h_f, bs_grad)
try:
    plt.axvline(HA.q1.history[-1])
except TypeError:
    pass
plt.axhline(0.0)

plt.figure()
plt.plot(l)

plt.show()
