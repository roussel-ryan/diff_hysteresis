import numpy as np

import bayes_hysteresis
import hysteresis
import synthetic
import torch
import matplotlib.pyplot as plt
from bayes_hysteresis_correlated import CorrelatedBayesianHysteresis
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, Trace_ELBO
import pyro
from bayesian_utils import train, predict
import utils
from data_imports import get_synthetic


# test fitting with hysteresis class
def main():
    n_grid = 25
    h, m, hyst_model = get_synthetic(n_grid)

    xx, yy = hyst_model.get_mesh()
    model = CorrelatedBayesianHysteresis(hyst_model,
                                         n_grid,
                                         0.1,
                                         use_prior=True)
    guide = AutoMultivariateNormal(model)

    train(h, m, model, guide, 20000, 0.001)
    summary = predict(h, model, guide)

    loc = pyro.param('AutoMultivariateNormal.loc')[:-2].double()
    raw_den = utils.vector_to_tril(loc, n_grid)
    den = utils.vector_to_tril(torch.nn.Softplus()(loc),
                               n_grid)

    y = summary['obs']
    fig, ax = plt.subplots()
    ax.plot(m.detach(), 'o')
    ax.plot(y['mean'].detach())
    ax.fill_between(range(len(h)),
                     y['5%'],
                     y['95%'],
                     alpha=0.25)

    fig, ax = plt.subplots()
    ax.plot(h, m.detach(), 'o')

    fig, ax = plt.subplots()
    c = ax.pcolor(xx, yy, den.detach())
    fig.colorbar(c)

    fig, ax = plt.subplots()
    c = ax.pcolor(xx, yy, raw_den.detach())
    fig.colorbar(c)


if __name__ == '__main__':
    main()
    plt.show()
