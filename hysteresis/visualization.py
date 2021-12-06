import matplotlib.pyplot as plt
import torch

from hysteresis.base import TorchHysteresis


def plot_hysteresis_density(H: TorchHysteresis, density=None):
    fig, ax = plt.subplots()
    x = H.mesh_points[:, 0]
    y = H.mesh_points[:, 1]

    if density is None:
        density = H.hysterion_density.detach()

    den = density  # * H.get_mesh_size(x, y)
    c = ax.tripcolor(x, y, den)
    fig.colorbar(c)
    return fig, ax


def plot_bayes_predicition(summary, m, baseline=False):
    y = summary["obs"]
    fig, ax = plt.subplots()
    ax.plot(m.detach(), "C1o", label="Data")

    mean = y["mean"]
    upper = y["mean"] + y["std"]
    lower = y["mean"] - y["std"]

    if isinstance(baseline, torch.Tensor):
        mean = mean - m
        upper = upper - m
        lower = lower - m
    ax.plot(mean, "C0", label="Model prediction")
    ax.fill_between(range(len(m)), upper, lower, alpha=0.25)
    ax.set_xlabel("step")
    ax.set_ylabel("B (arb. units)")
    ax.legend()

    return fig, ax
