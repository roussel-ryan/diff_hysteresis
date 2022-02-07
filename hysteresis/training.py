import logging
from copy import deepcopy

import torch

from hysteresis.modes import FITTING

logger = logging.getLogger(__name__)


def train_MSE(model: torch.nn.Module, train_x, train_y, n_steps, lr=0.1, atol=1.0e-8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_track = []
    best_loss = 1e10
    best_state = deepcopy(model.state_dict())
    for i in range(n_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = torch.nn.MSELoss()(train_y, output)
        loss.backward()

        loss_track += [loss]
        min_loss = torch.min(torch.tensor(loss_track))
        if min_loss < atol:
            break

        if min_loss < best_loss:
            best_state = deepcopy(model.state_dict())
            best_loss = min_loss

        optimizer.step()
        if i % 1000 == 0:
            print(i)

    model.load_state_dict(best_state)
    return torch.tensor(loss_track)


def train_hysteresis(model, n_steps, lr=0.1, atol=1e-8):
    model.mode = FITTING
    train_x = model.history_h
    train_y = model.transformer.transform(model.history_h, model.history_m)[1]

    return train_MSE(model, train_x, train_y, n_steps, lr=lr, atol=atol)
