from hysteresis.densities import linear
from hysteresis.visualization import plot_hysteresis_density
import torch
import matplotlib.pyplot as plt
from hysteresis.base import TorchHysteresis


class TestDensities:
    def test_densities(self):
        H = TorchHysteresis(mesh_scale=0.1)
        H.hysterion_density = linear(H.mesh_points)

        h_test = torch.linspace(0, 1, 50)
        m = H.predict_magnetization(h_test)

        fig, ax = plt.subplots()
        ax.plot(h_test, m.detach())

        plot_hysteresis_density(H)
        plt.show()
