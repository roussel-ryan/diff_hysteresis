import bayes_hysteresis
import synthetic
import torch
import matplotlib.pyplot as plt
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, Trace_ELBO
import pyro

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
    n_grid = 10

    h_max = 0.5
    h_min = -h_max
    b_sat = 1.0

    # get synthetic training h_data
    h, m = synthetic.generate_saturation_dataset(15, n_grid, h_max, b_sat)

    # scale m to be reasonable
    m = m / max(m)

    h = h[:15]
    m = m[:15]

    model = bayes_hysteresis.BayesianHysteresis(h, h_min, h_max, b_sat, n_grid)
    guide = AutoDiagonalNormal(model)

    adam = pyro.optim.Adam({"lr": 0.01})
    svi = SVI(model, guide, adam, loss=TraceEnum_ELBO())

    # plot model before training
    predictive = Predictive(model, guide=guide, num_samples=100,
                            return_sites=['_RETURN'])

    samples = predictive(h)
    pred_summary = summary(samples)
    mu = pred_summary['_RETURN']
    print(mu.keys())

    fig, ax = plt.subplots()
    ax.plot(h, m.detach(), 'o')
    ax.plot(h, mu['mean'].detach())
    ax.fill_between(h,
                    mu['5%'],
                    mu['95%'],
                    alpha=0.25)

    pyro.clear_param_store()
    num_iterations = 500
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(h, m)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss))

    predictive = Predictive(model, guide=guide, num_samples=100,
                            return_sites=['_RETURN'])

    samples = predictive(h)
    pred_summary = summary(samples)
    mu = pred_summary['_RETURN']

    fig, ax = plt.subplots()
    ax.plot(h, m.detach(), 'o')
    ax.plot(h, mu['mean'].detach())
    ax.fill_between(h,
                    mu['5%'],
                    mu['95%'],
                    alpha=0.25)


if __name__ == '__main__':
    main()
    plt.show()
