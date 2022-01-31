import torch
from matplotlib import pyplot as plt

from hysteresis.base import BaseHysteresis
from hysteresis.reconstruction import reconstruction
from hysteresis.visualization import plot_hysterion_density
from hysteresis.training import train_hysteresis


class TestReconstruction:
    def test_reconstruction(self):
        train_h = torch.linspace(0, 20, 10)
        train_m = 2.0 * train_h
        H = BaseHysteresis(train_h, train_m, mesh_scale=0.5)
        H.scale = 0
        print(H.hysterion_density)
        print(H.offset)
        print(H.slope)
        print(H.transformer)

        H_new = reconstruction(H)

        print(H_new.hysterion_density)
        print(H_new.offset)
        print(H_new.slope)
        print(H.transformer.scale_m)

        test_h = train_h.clone()
        H_new.regression()
        pred_m = H_new(test_h, return_real=True)

        plot_hysterion_density(H_new)
        plt.figure()
        plt.plot(test_h, pred_m.detach())
        plt.plot(train_h, train_m)
        plt.show()
