import matplotlib.pyplot as plt
import pyro
import torch
from pyro.infer.autoguide import AutoDiagonalNormal

import bayes_hysteresis
import data_imports
import utils
from bayesian_utils import train, predict
from plotting import plot_bayes_predicition


# test fitting with hysteresis class
def main():
    n_grid = 25
    h, m, model = data_imports.get_real(n_grid)

    model = bayes_hysteresis.BayesianHysteresis(model, n_grid)
    guide = AutoDiagonalNormal(model)

    train(h, m, model, guide, 5000, 0.01)
    summary = predict(h, model, guide)

    loc = pyro.param('AutoDiagonalNormal.loc')[:-2].double()
    den = utils.vector_to_tril(torch.nn.Softplus()(loc),
                               n_grid)

    y = summary['obs']
    fig, ax = plot_bayes_predicition(summary, m)
    fig.savefig('figures/bayes_prediction_real.svg')

    # fitted density
    loc = pyro.param('AutoDiagonalNormal.loc')[:-2].double()
    scale = pyro.param('AutoDiagonalNormal.scale')[:-2].double()

    den = utils.vector_to_tril(torch.nn.Softplus()(loc),
                               n_grid)
    upper = utils.vector_to_tril(torch.nn.Softplus()(loc + scale), n_grid)
    lower = utils.vector_to_tril(torch.nn.Softplus()(loc - scale), n_grid)

    xx, yy = model.hysteresis_model.get_mesh()
    xx = xx.numpy()
    yy = yy.numpy()

    fig, ax = plt.subplots()
    c = ax.pcolor(xx, yy, den.detach().numpy())
    fig.colorbar(c, label='Hysterion Density (arb. units)')
    ax.set_xlabel(r'$\beta$ (A)')
    ax.set_ylabel(r'$\alpha$ (A)')
    fig.savefig('figures/bayes_mean_density_real.png', dpi=300)


if __name__ == '__main__':
    main()
    plt.show()
