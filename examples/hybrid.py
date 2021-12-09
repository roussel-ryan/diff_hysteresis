import matplotlib.pyplot as plt
import torch

from hysteresis.base import BaseHysteresis
from hysteresis.hybrid import HybridGP
from hysteresis.visualization import plot_hysteresis_density


def density_function(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / 0.1)

# generate some dummy data
train_x = torch.linspace(-10.0, 10.0, 10)

H = BaseHysteresis(train_x, trainable=False, mesh_scale=0.1, temp=1e-3)
H.hysterion_density = density_function(H.mesh_points)

plot_hysteresis_density(H, H._states[-1])

train_m = H(train_x).detach()
train_y = train_m ** 2

plt.figure()
plt.plot(train_x, train_m)

hgp = HybridGP(train_x.reshape(-1, 1), train_y.reshape(-1, 1), H)
hgp.train_model()

# make initial predictions
test_x = torch.linspace(-10.0, 10.0, 50).reshape(-1, 1)
with torch.no_grad():
    post = hgp(test_x, untransform_posterior=True)
    mean = post.mean.flatten()
    std = torch.sqrt(post.variance).flatten()

fig, ax = plt.subplots()
ax.plot(train_x, train_y.detach(), 'o')
ax.plot(test_x, mean, 'C1')
ax.fill_between(test_x.flatten(), mean - std, mean + std, alpha=0.25, fc='C1')
plt.show()

