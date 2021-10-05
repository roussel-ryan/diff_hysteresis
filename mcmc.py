from pyro.infer import MCMC, NUTS


def run_mcmc(model, data, num_samples=500):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples)

    mcmc.run(*data)

    return {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}


