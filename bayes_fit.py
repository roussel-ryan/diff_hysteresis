import bayes_hysteresis
import hysteresis
import synthetic
import torch
import matplotlib.pyplot as plt
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, Trace_ELBO
import pyro
from bayesian_utils import train, predict
import utils
from plotting import plot_bayes_predicition

def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


# test fitting with hysteresis class
def main():
    n_grid = 25

    h_max = 0.5
    h_min = -h_max
    b_sat = 1.0

    # get synthetic training h_data
    h, m = synthetic.generate_saturation_dataset(15, n_grid, h_max, b_sat)

    h = h.detach().double()
    m = m.detach()

    # scale m to be reasonable
    m = m / max(m)

    # h = h[:15]
    # m = m[:15]

    model = hysteresis.Hysteresis(h,
                                  h_min,
                                  h_max,
                                  b_sat,
                                  n_grid,
                                  trainable=False)

    model = bayes_hysteresis.BayesianHysteresis(model, n_grid)
    guide = AutoDiagonalNormal(model)

    train(h, m, model, guide, 5000, 0.01)
    summary = predict(h, model, guide)

    loc = pyro.param('AutoDiagonalNormal.loc')[:-2].double()
    den = utils.vector_to_tril(torch.nn.Softplus()(loc),
                               n_grid)

    y = summary['obs']
    fig, ax = plot_bayes_predicition(summary, m)
    fig.savefig('figures/bayes_prediction.svg')

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
    ax.set_xlabel(r'$\beta$ (T)')
    ax.set_ylabel(r'$\alpha$ (T)')
    fig.savefig('figures/bayes_mean_density.png', dpi=300)

if __name__ == '__main__':
    main()
    plt.show()
