import torch
import hysteresis
import utils


def gaussian_density(xx, yy):
    x = xx.flatten()
    y = yy.flatten()
    pts = torch.hstack([ele.reshape(-1, 1) for ele in [x, y]])

    m = torch.distributions.MultivariateNormal(
        torch.zeros(2).double(), 0.1 * torch.eye(2).double()
    )
    dens = torch.exp(m.log_prob(pts))
    # print(dens.shape)

    return dens.reshape(xx.shape[0], xx.shape[1])


def generate_saturation_dataset(n_data, n_mesh, h_sat, b_sat):
    h_sub = torch.linspace(-h_sat, h_sat, n_data)
    h = torch.cat((h_sub, torch.flip(h_sub, [0])))

    H = hysteresis.Hysteresis(h, -h_sat, h_sat, b_sat, n_mesh)
    xx, yy = H.get_mesh()

    synthetic_mu = gaussian_density(xx, yy)
    H.set_density_vector(utils.tril_to_vector(synthetic_mu, n_mesh))
    b = H.predict_magnetization()

    return h, b


def generate_one_sided_dataset(n_data, n_mesh, h_sat, b_sat):
    h_sub = torch.linspace(0.01, h_sat, n_data)
    h_first_loop = torch.cat((h_sub, torch.flip(h_sub, [0])))
    h_second_loop = torch.cat((h_sub, torch.flip(h_sub, [0])))
    h = torch.cat((h_first_loop, h_second_loop, -h_sub))

    H = hysteresis.Hysteresis(h, -h_sat, h_sat, b_sat, n_mesh)
    xx, yy = H.get_mesh()

    synthetic_mu = gaussian_density(xx, yy)
    H.set_density_vector(utils.tril_to_vector(synthetic_mu, n_mesh))
    b = H.predict_magnetization()

    return h, b
