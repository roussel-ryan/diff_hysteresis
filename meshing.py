import pygmsh
import numpy as np


def create_mesh(size):
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [-1, -1],
                [1, 1],
                [-1, 1],
            ],
            mesh_size=0.1
        )

        # set mesh size with function
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z: (0.2 * (np.abs(x - y)) + 0.05) * size
        )

        mesh = geom.generate_mesh()

    return mesh.points[:-1]
