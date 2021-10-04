import numpy as np

import bayes_hysteresis
import hysteresis
import synthetic
import torch
import matplotlib.pyplot as plt
from bayes_hysteresis_correlated import calculate_distances, \
    CorrelatedBayesianHysteresis
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, Trace_ELBO
import pyro
from bayesian_utils import train, predict
import utils

# test fitting with hysteresis class
def main():
    n_grid = 25

    h_max = 1.0
    h_min = 0.0
    b_sat = 1.0

    # get real h, m
    data = torch.tensor(np.loadtxt('data/argonne_data.txt'))
    h = data.T[0]
    m = data.T[1]

    # normalize h, m
    h = (h - min(h)) / (max(h) - min(h))
    m = (m - min(m)) / (max(m) - min(m))

    h = h.detach().double()
    m = m.detach()

    # scale m to be reasonable
    m = m / max(m)

    # h = h[:15]
    # m = m[:15]

    hyst_model = hysteresis.Hysteresis(h,
                                       h_min,
                                       h_max,
                                       b_sat,
                                       n_grid,
                                       trainable=False)
    xx, yy = hyst_model.get_mesh()
    model = CorrelatedBayesianHysteresis(hyst_model,
                                         n_grid, 0.1)
    guide = AutoMultivariateNormal(model)

    train(h, m, model, guide, 10000, 0.001)
    summary = predict(h, model, guide)

    loc = pyro.param('AutoMultivariateNormal.loc')[:-2].double()
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

if __name__ == '__main__':
    main()
    plt.show()
