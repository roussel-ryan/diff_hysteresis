import torch
from pyro.infer import Predictive


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


def predict(h, model, guide):
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=500,
        return_sites=["_RETURN", "obs", "density", "scale", "offset"],
    )

    samples = predictive(h)
    pred_summary = summary(samples)
    return pred_summary, samples
