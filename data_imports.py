import hysteresis
import synthetic
import torch
import numpy as np


def get_synthetic(n_grid=25, cuda=False, train_index=-1):
    h_max = 0.5
    h_min = -h_max
    b_sat = 1.0

    # get synthetic training h_data
    h, m = synthetic.generate_saturation_dataset(15, n_grid, h_max, b_sat)

    h = h.detach()
    m = m.detach()

    # scale m to be reasonable
    m = m / max(m)

    if train_index > 0:
        h = h[:train_index]
        m = m[:train_index]

    if cuda:
        h = h.cuda()
        m = m.cuda()

    model = hysteresis.Hysteresis(h,
                                  h_min,
                                  h_max,
                                  b_sat,
                                  n_grid,
                                  trainable=False)
    if cuda:
        model = model.cuda()

    return h, m, model


def get_real(n_grid):
    h_max = 200
    h_min = 175
    b_sat = 1.0

    # get real h, m
    data = torch.tensor(np.loadtxt('data/argonne_data.txt'))
    h = data.T[0]
    m = data.T[1]

    h = h.detach().double()
    m = m.detach()

    # scale m to be reasonable
    m = (m - min(m)) / (max(m) - min(m))

    model = hysteresis.Hysteresis(h,
                                  h_min,
                                  h_max,
                                  b_sat,
                                  n_grid,
                                  trainable=False)
    return h, m, model
