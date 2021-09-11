import torch
import numpy as np
import numerical_hysteresis
import utils

def gaussian_density(xx, yy):

    x = xx.flatten()
    y = yy.flatten()
    pts = torch.hstack([ele.reshape(-1, 1) for ele in [x, y]])

    m = torch.distributions.MultivariateNormal(torch.zeros(2).double(), 0.1*torch.eye(2).double())
    dens = torch.exp(m.log_prob(pts))
    #print(dens.shape)

    return dens.reshape(xx.shape[0], xx.shape[1])

def generate_dataset(n_data):
    h = np.append(np.linspace(-1.0, 1.0, n_data), np.flipud(np.linspace(-1.0, 1.0, n_data)))
    n = 100
    h_sat = 1.0
    xx, yy = utils.generate_mesh(h_sat, n)
    synthetic_mu = gaussian_density(xx, yy)
    states = numerical_hysteresis.state(xx, yy, h_sat, h)
    mu_vector = utils.tril_to_vector(synthetic_mu, n)
    b = numerical_hysteresis.discreteIntegral(xx, yy, 0.8, 2, mu_vector, h,n, states)
    
    return h, b



