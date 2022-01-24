import torch
from .base import BaseHysteresis


def reconstruction(H_model: BaseHysteresis):
    # get grid points on alpha=beta line
    mesh_points = H_model.mesh_points

    boundary_indicies = torch.nonzero(mesh_points[:,0] == mesh_points[:,1]).squeeze()
    boundary_pts = mesh_points[boundary_indicies]

    true_hysterion_density = torch.zeros(len(mesh_points))
    true_hysterion_density[boundary_indicies] = H_model.slope

    H = BaseHysteresis()
    H.scale = 1.0
    H.offset = 0.0
    H.slope = 0.0

    H.hysterion_density = true_hysterion_density
    return H
