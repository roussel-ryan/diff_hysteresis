import pygmsh
import numpy as np
import matplotlib.pyplot as plt


def constant_mesh_size(x, y, mesh_scale):
    return mesh_scale


def default_mesh_size(x, y, mesh_scale):
    return mesh_scale * (0.2 * (np.abs(x - y)) + 0.05)


def exponential_mesh(x, y, mesh_scale, min_density=0.001, ls=0.05):
    return mesh_scale * (1.0 - np.exp(-np.abs(x - y) / ls)) + min_density


def create_triangle_mesh(mesh_scale, mesh_density_function=None):
    mesh_density_function = mesh_density_function or constant_mesh_size
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [0, 0],
                [1, 1],
                [0, 1],
            ],
            mesh_size=0.1,
        )

        # set mesh size with function
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z: mesh_density_function(x, y, mesh_scale)
        )

        mesh = geom.generate_mesh()

    return mesh.points[:, :-1]


if __name__ == "__main__":
    t = np.linspace(0, 0.5)
    x = 0.5 - t
    y = 0.5 + t
    ms = 1.0
    plt.plot(x, y)
    plt.figure()
    plt.plot(t, default_mesh_size(x, y, ms))
    plt.plot(t, exponential_mesh(x, y, ms))
    plt.show()
