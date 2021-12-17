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


def predict(h, model, guide, num_samples=500):
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=num_samples,
        return_sites=[
            "_RETURN",
            "obs",
            "raw_hysterion_density",
            "scale",
            "offset",
            "slope",
        ],
    )

    samples = predictive(h, return_real=True)
    # samples["density"] = model.raw_hysterion_density.constraint.inverse_transform(
    #    samples["raw_hysterion_density"]
    # )
    # del samples["raw_hysterion_density"]
    pred_summary = summary(samples)
    return pred_summary, samples


def save_predictive(model, guide, num_samples=500):
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=num_samples,
        return_sites=[
            "_RETURN",
            "obs",
            "_raw_hysterion_density",
            "scale",
            "offset",
            "slope",
        ],
    )
    torch.save(predictive, "predictive.pt")


def load_predictive(fname):
    return torch.load(fname)
