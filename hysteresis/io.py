import pyro.nn.module
from pyro.infer.autoguide import AutoDelta, AutoNormal
import torch
from .base import BaseHysteresis
from .bayesian import BayesianHysteresis


def load_base(fname):
    model = BaseHysteresis()
    model.load_state_dict(torch.load(fname))
    return model


def save_base(model: BaseHysteresis, fname: str):
    torch.save(model.state_dict(), fname)


