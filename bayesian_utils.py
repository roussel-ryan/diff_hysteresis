import torch
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, Trace_ELBO
import pyro


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


def train(h, m, model, guide, num_steps, initial_lr=0.001, gamma=0.1):
    lrd = gamma ** (1 / num_steps)
    optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    pyro.clear_param_store()
    for j in range(num_steps):
        # calculate the loss and take a gradient step
        loss = svi.step(h, m)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss))


def predict(h, model, guide):
    predictive = Predictive(model,
                            guide=guide,
                            num_samples=500,
                            return_sites=['_RETURN', 'obs', 'density'])

    samples = predictive(h)
    pred_summary = summary(samples)
    return pred_summary, samples
