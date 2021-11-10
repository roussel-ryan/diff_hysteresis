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


# hysteresis model
H = TorchHysteresis(mesh_scale=0.1, trainable=False, temp=5e-3)
dens = density_function(H.mesh_points)
H.h_min = -1.0
H.hysterion_density = dens

# define quadrupoles
q1 = HysteresisQuad("q1", torch.tensor(0.1), deepcopy(H), scale=torch.tensor(100.0))
d1 = TorchDrift("d1", torch.tensor(1.0))


HA = HysteresisAccelerator([q1, d1])

R = torch.eye(6)

# predict transport matrix of quad at negative limit

# predict the beam matrix at negative limit
print(HA(R))

# print named parameters
for name, val in HA.named_parameters():
    print(f"{name}:{val}")

# plot beam size as a function of fantasy_H
h_f = torch.linspace(-1, 1, 100)
bs = []
bs_grad = []
for ele in h_f:

    HA.q1.fantasy_H.data = ele
    bs += [HA(R)[0, 0].detach()]

    HA(R)[0, 0].backward()
    bs_grad += [HA.q1.fantasy_H.grad.clone()]
    HA.q1.fantasy_H.grad.zero_()

plt.plot(h_f, bs_grad)

plt.figure()
plt.plot(h_f, bs, label="step 0")

# then apply a field and re-plot vs h_fantasy
HA.apply_fields({"q1": torch.ones(1) * 0.5})

h_f = torch.linspace(-1, 1, 100)
bs = []
for ele in h_f:
    HA.q1.fantasy_H.data = torch.atleast_1d(ele)
    bs += [HA(R)[0, 0].detach()]

    # calculate derivative


plt.plot(h_f, bs, label="step 1")

# then apply a field and re-plot vs h_fantasy
HA.apply_fields({"q1": torch.ones(1) * -0.5})

h_f = torch.linspace(-1, 1, 100)
bs = []
for ele in h_f:
    HA.q1.fantasy_H.data = torch.atleast_1d(ele)
    bs += [HA(R)[0, 0].detach()]

plt.plot(h_f, bs, label="step 2")
plt.legend()
plt.ylabel("Beam size")
plt.xlabel("H fantasy")
plt.show()
