import gpytorch.likelihoods
import matplotlib.pyplot as plt
import torch
from gpytorch.likelihoods import GaussianLikelihood

from hysteresis.training import train_MSE
from hysteresis.base import BaseHysteresis
from hysteresis.hybrid import ExactHybridGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

aps_model = torch.load("aps_model.pt")

train_h = aps_model.history_h
train_m = aps_model.history_m

train_y = torch.sin(train_m * 2 * 3.14 / 40)
# plt.plot(train_h, train_m)
# plt.figure()
# plt.plot(train_h, train_y)

# hysteresis model
H = BaseHysteresis(train_h)
hgp = ExactHybridGP(train_h.reshape(-1, 1), train_y.reshape(-1, 1), H)

mll = ExactMarginalLogLikelihood(hgp.gp.likelihood, hgp)
print(list(mll.named_parameters()))

fit_gpytorch_model(mll)
