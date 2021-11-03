import torch
from hysteresis.base import TorchHysteresis
from hysteresis.visualization import plot_hysteresis_density

class TestVisualization:
    def test_plot_density(self):
        h_data = torch.rand(10) * 10.0
        H = TorchHysteresis(h_data)
        plot_hysteresis_density(H)
