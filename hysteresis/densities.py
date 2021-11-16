import torch


def linear(mesh_pts):
    x = mesh_pts[:, 0]
    y = mesh_pts[:, 1]
    return torch.max(torch.tensor(x == y), torch.tensor(0.01))
