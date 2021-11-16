import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from hysteresis.base import TorchHysteresis
from hysteresis.hybrid import HybridGP


def density_function(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.exp(-(y - x) / 0.1)


class TestHysteresisGP:
    def test_training_1D(self):
        # generate some dummy data
        train_x = torch.linspace(-1, 1, 7).double()

        H = TorchHysteresis(train_x, trainable=False, mesh_scale=0.1, h_min=-1.0)
        H.hysterion_density = density_function(H.mesh_points)

        train_m = H.predict_magnetization().detach()
        train_y = train_m ** 2

        hgp = HybridGP(train_x.reshape(-1, 1), train_y.reshape(-1, 1), H)
        hgp.train_model()

        # for name, item in hgp.named_parameters():
        #     print(f'{name}:{item}')

        # make initial predictions
        test_x = torch.linspace(-1.0, 1.0, 50).reshape(-1, 1)
        with torch.no_grad():
            post = hgp(test_x)
            mean = post.mean.flatten()
            std = torch.sqrt(post.variance).flatten()

        # fig, ax = plt.subplots()
        # ax.plot(train_x, train_y.detach(), 'o')
        # ax.plot(test_x, mean, 'C1')
        # ax.fill_between(test_x, mean - std, mean + std, alpha=0.25, fc='C1')
        # plt.show()

    def test_training_ND(self):
        # generate some dummy data
        train_x = torch.cat([torch.linspace(-1.0, 1.0, 10).reshape(-1, 1)] * 3, dim=1)

        H1 = TorchHysteresis(trainable=False, mesh_scale=0.1, h_min=-1.0)
        H1.hysterion_density = density_function(H1.mesh_points)

        models = [H1, deepcopy(H1), deepcopy(H1)]
        train_m = []
        for idx, hyst_model in enumerate(models):
            hyst_model.applied_fields = train_x[:, idx]
            train_m += [hyst_model.predict_magnetization()]
        train_m = torch.cat([ele.unsqueeze(1) for ele in train_m], dim=1)

        train_y = train_m[:, 0] ** 2 + train_m[:, 1] ** 2 + train_m[:, 2] ** 2

        hgp = HybridGP(train_x, train_y.reshape(-1, 1), models)
        hgp.train_model()

        # for name, item in hgp.named_parameters():
        #     print(f'{name}:{item}')

        test_x = torch.cat([torch.linspace(-1.0, 1.0, 50).reshape(-1, 1)] * 3, dim=1)
        with torch.no_grad():
            post = hgp(test_x)
            mean = post.mean.flatten()
            std = torch.sqrt(post.variance).flatten()

        # fig, ax = plt.subplots()
        # ax.plot(train_x[:, 0], train_y.detach(), 'o')
        # ax.plot(test_x[:, 0], mean, 'C1')
        # plt.show()
