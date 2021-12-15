import torch
from hysteresis.base import BaseHysteresis
from hysteresis.visualization import plot_hysterion_density


class TestVisualization:
    def test_plot_density(self):
        h_data = torch.rand(10) * 10.0
        H = BaseHysteresis(h_data)
        plot_hysterion_density(H)
