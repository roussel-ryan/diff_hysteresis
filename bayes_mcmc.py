import matplotlib.pyplot as plt
import pyro
import torch

import bayes_hysteresis
import data_imports
import utils
import bayesian_utils
import mcmc
from plotting import plot_bayes_predicition

torch.set_default_tensor_type(torch.cuda.DoubleTensor)


# test fitting with hysteresis class
def main():
    n_grid = 4
    use_cuda = True

    h, m, model = data_imports.get_synthetic(n_grid, cuda=use_cuda)

    if use_cuda:
        h = h.cuda()
        m = m.cuda()
        model = model.cuda()

    bayes_model = bayes_hysteresis.BayesianHysteresis(model, n_grid)
    hmc_samples = mcmc.run_mcmc(bayes_model, [h, m])


if __name__ == '__main__':
    main()
    plt.show()
