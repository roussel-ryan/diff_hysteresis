# %%
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

# %%

n_grid = 25

# %%
models = []
sums = []
means = []
guides = []
for state in [True, False]:
    h, m, hyst_model = get_synthetic(n_grid)
    train_h = h[:10]
    train_m = m[:10]
    hyst_model.set_data(train_h)
    model = CorrelatedBayesianHysteresis(hyst_model,
                                         n_grid,
                                         0.1,
                                         use_prior=state)

    guide = AutoMultivariateNormal(model)
    train(train_h, train_m, model, guide, 10000, 0.001)

    models += [model]
    guides += [guide]

# %%
xx, yy = models[0].hysteresis_model.get_mesh()
test_h = torch.linspace(-1, 1, 50)
titles = ['w/ prior', 'w/o prior']
for i in [0, 1]:
    summary = predict(h, models[i], guides[i])
    den = utils.vector_to_tril(
        torch.nn.Softplus()(summary['density']['mean'].flatten()), n_grid)
    fig, ax = plt.subplots()
    c = ax.pcolor(xx, yy, den.detach())
    ax.set_title(titles[i])
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\alpha$')
    fig.colorbar(c)

    fig2, ax2 = plt.subplots()
    y = summary['obs']
    ax2.plot(m.detach(), 'o')
    ax2.plot(train_m.detach(), 'o')

    ax2.plot(y['mean'])
    ax2.fill_between(range(len(h)),
                     y['5%'],
                     y['95%'],
                     alpha=0.25)
    ax2.set_title(titles[i])
    ax2.set_xlabel('H')
    ax2.set_ylabel('M')

plt.show()
# %%
for name, item in pyro.get_param_store().items():
    print(f'{name}:{item}')

out = model(test_h)
print(out)
# %%
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
# %%
