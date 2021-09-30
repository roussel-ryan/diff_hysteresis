import bayes_hysteresis
import hysteresis
import synthetic
import torch
import matplotlib.pyplot as plt
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, Trace_ELBO
import pyro
import numpy as np

import utils


def loss_fn(m, m_pred):
    return torch.sum((m - m_pred) ** 2)


def train(model, m, n_steps, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)

    loss_track = []
    for i in range(n_steps):
        optimizer.zero_grad()
        output = model.predict_magnetization()
        loss = loss_fn(m, output)
        loss.backward(retain_graph=True)

        loss_track += [loss]
        optimizer.step()
        if i % 100 == 0:
            print(i)

    return torch.tensor(loss_track)


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

    h_max = 1
    h_min = 0
    b_sat = h_max

    # get real h, m
    data = torch.tensor(np.loadtxt('data/argonne_data.txt'))
    h = data.T[0]
    m = data.T[1]

    # normalize h, m
    h = (h - min(h)) / (max(h) - min(h))
    m = (m - min(m)) / (max(m) - min(m))

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

    num_steps = 5000
    initial_lr = 0.01
    gamma = 1.0  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / num_steps)
    optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    pyro.clear_param_store()
    for j in range(num_steps):
        # calculate the loss and take a gradient step
        loss = svi.step(h, m)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss))

    predictive = Predictive(model,
                            guide=guide,
                            num_samples=500,
                            return_sites=['_RETURN', 'obs'])

    samples = predictive(h)
    pred_summary = summary(samples)
    mu = pred_summary['obs']

    fig, ax = plt.subplots()
    ax.plot(h, m.detach(), 'o')
    ax.plot(h, mu['mean'].detach())
    ax.fill_between(h,
                    mu['5%'],
                    mu['95%'],
                    alpha=0.25)

    fig3, ax3 = plt.subplots()
    ax3.plot(m.detach(), 'o')
    ax3.plot(mu['mean'].detach())
    ax3.fill_between(range(len(h)),
                     mu['5%'],
                     mu['95%'],
                     alpha=0.25)

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

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, den.detach().numpy(),
                    linewidth=0)
    ax.plot_surface(xx, yy, upper.detach().numpy())
    ax.plot_surface(xx, yy, lower.detach().numpy())

    fig, ax = plt.subplots()
    c = ax.pcolor(xx, yy, den.detach().numpy())
    fig.colorbar(c)


if __name__ == '__main__':
    main()
    plt.show()
