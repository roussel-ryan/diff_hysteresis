import torch
from matplotlib import pyplot as plt

from hysteresis.base import BaseHysteresis
from hysteresis.reconstruction import reconstruction
from hysteresis.visualization import plot_hysterion_density


class TestReconstruction:
    def test_reconstruction(self):
        H = BaseHysteresis()
        print(H.hysterion_density)
        print(H.offset)
        print(H.slope)
        print(H.transformer)

        H_new = reconstruction(H)

        test_h = torch.linspace(0, 1, 20)
        H_new.regression()
        pred_m = H_new(test_h)

        plot_hysterion_density(H_new)
        plt.figure()
        plt.plot(test_h, pred_m.detach())
        plt.show()
