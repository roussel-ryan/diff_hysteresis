import matplotlib.pyplot as plt
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDelta, AutoNormal

import logging

logger = logging.getLogger(__name__)


def train_MSE(model, train_x, train_y, n_steps, lr=0.1, atol=1.0e-8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    loss_track = []
    for i in range(n_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = torch.nn.MSELoss()(train_y, output)
        loss.backward()

        if loss < atol:
            break

        loss_track += [loss]
        optimizer.step()
        if i % 1000 == 0:
            logger.debug(i)

    return torch.tensor(loss_track)


def train_bayes(h, m, model, num_steps, guide=None, initial_lr=0.001, gamma=0.1):
    guide = guide or AutoNormal(model)

    lrd = gamma ** (1 / num_steps)
    optim = pyro.optim.ClippedAdam({"lr": initial_lr, "lrd": lrd})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    pyro.clear_param_store()
    loss_trace = []
    for j in range(num_steps):
        # calculate the loss and take a gradient step
        loss = svi.step(h, m)
        loss_trace += [loss]
        if j % 100 == 0:
            logger.debug("[iteration %04d] loss: %.4f" % (j + 1, loss))

    return guide, torch.tensor(loss_trace)


def map_bayes(h, m, model, num_steps, initial_lr=0.001, gamma=0.1):
    """maximum a posteriori point estimation of parameters"""
    guide = AutoDelta(model)

    return train_bayes(h, m, model, num_steps, guide, initial_lr, gamma)


def mle_bayes(h, m, model, num_steps, initial_lr=0.001, gamma=0.1):
    def empty_guide(X, Y):
        pass

    return train_bayes(h, m, model, num_steps, empty_guide, initial_lr, gamma)
