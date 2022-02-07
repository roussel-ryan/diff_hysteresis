import torch
from .base import BaseHysteresis


def reconstruction(H_model: BaseHysteresis):
    # get grid points on alpha=beta line
    mesh_points = H_model.mesh_points

    boundary_indicies = torch.nonzero(mesh_points[:, 0] == mesh_points[:, 1]).squeeze()
    boundary_pts = mesh_points[boundary_indicies]

    # calculate the gradient of the transformer @ the boundary points
    poly_grad = H_model.transformer.get_fit_grad(boundary_pts[:, 0])

    true_hysterion_density = torch.zeros(len(mesh_points)).double()
    polynmoial_contrib = true_hysterion_density.clone()
    polynmoial_contrib[boundary_indicies] = (
        (H_model.slope * H_model.transformer.scale_m + poly_grad)
        * torch.sqrt(torch.tensor(2.0))
        / len(boundary_pts)
    )
    old_density_contrib = H_model.scale * H_model.hysterion_density

    true_hysterion_density += polynmoial_contrib + old_density_contrib

    print(torch.sum(true_hysterion_density[: len(boundary_pts)]))
    H = BaseHysteresis(
        mesh_scale=H_model.mesh_scale,
        fixed_domain=H_model.valid_domain,
        use_normalized_density=False,
    )
    H.offset = (
        H_model.offset * H_model.transformer.scale_m + H_model.transformer.offset_m
    )
    H.slope = 0.0

    H.hysterion_density = true_hysterion_density
    H.scale = torch.sum(H.hysterion_density)

    return H
