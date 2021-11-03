from hysteresis.meshing import create_triangle_mesh


class TestTriangleMesh:
    def test_triangle_mesh(self):
        mesh = create_triangle_mesh(1.0)
        print(len(mesh))
