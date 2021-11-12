import matplotlib.pyplot as plt
from botorch.models.transforms import Standardize

from hysteresis.base import TorchHysteresis
from torchAccelerator.hysteresis import HysteresisAccelerator, HysteresisQuad
from torchAccelerator.first_order import TorchDrift
import torch
from copy import deepcopy

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf


def density_function(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / 0.5)


def beamsize(accel_model, R_i):
    return accel_model.forward(R_i)[0, 0]


def total_beamsize(R):
    return R[0, 0] + R[2, 2]


def optimize(accelerator_model, initial_beam_matrix, apply_fields=False):

    iterations = 50
    train_X = torch.ones((1, 3)) * -0.25
    accelerator_model.q1.fantasy_H.data = train_X[0][0]
    accelerator_model.q2.fantasy_H.data = train_X[0][1]
    accelerator_model.q3.fantasy_H.data = train_X[0][2]

    train_Y = total_beamsize(
        accelerator_model.forward(initial_beam_matrix)
    ).reshape(1, 1)

    if apply_fields:
        print(f"applying field {train_X[-1]}")
        accelerator_model.apply_fields({"q1": train_X[-1][0],
                                        "q2": train_X[-1][1],
                                        "q3": train_X[-1][2]})

    for i in range(iterations):
        std_trans = Standardize(1)
        gp = SingleTaskGP(train_X, train_Y.detach(), outcome_transform=std_trans)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False)

        bounds = torch.stack([-1.0 * torch.ones(3), torch.ones(3)])
        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        train_X = torch.cat((train_X, candidate))
        accelerator_model.q1.fantasy_H.data = candidate[0][0]
        accelerator_model.q2.fantasy_H.data = candidate[0][1]
        accelerator_model.q3.fantasy_H.data = candidate[0][2]

        # make next measurement
        bs = total_beamsize(
            accelerator_model.forward(initial_beam_matrix)
        ).reshape(1, 1)
        train_Y = torch.cat((train_Y, bs))

        if apply_fields:
            print(f"applying field {train_X[-1]}")
            accelerator_model.apply_fields({"q1": train_X[-1][0],
                                            "q2": train_X[-1][1],
                                            "q3": train_X[-1][2]})

    return train_X, train_Y, gp


# hysteresis model
H = TorchHysteresis(mesh_scale=0.1, trainable=False)
dens = density_function(H.mesh_points)
H.h_min = -1.0
H.hysterion_density = dens

# define quadrupoles
q1 = HysteresisQuad("q1", torch.tensor(0.1), deepcopy(H), scale=torch.tensor(200.0))
d1 = TorchDrift("d1", torch.tensor(1.0))
q2 = HysteresisQuad("q2", torch.tensor(0.1), deepcopy(H), scale=torch.tensor(200.0))
q3 = HysteresisQuad("q3", torch.tensor(0.1), deepcopy(H), scale=torch.tensor(200.0))


HA = HysteresisAccelerator([q1, d1, q2, d1, q3, d1])

for name, val in HA.named_parameters():
    if val.requires_grad:
        print(f"{name}:{val}")

# do optimization
R = torch.eye(6)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

# plot beam size as a function of fantasy_H
h_f = torch.linspace(-1.0, 1, 100)

for ele, c in zip([False, True], ["C0", "C1"]):
    p, l, model = optimize(deepcopy(HA), R, ele)
    ax2.plot(l.squeeze().detach(), c=c)

print(p)

plt.show()
