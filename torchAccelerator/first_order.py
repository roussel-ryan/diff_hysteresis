import numpy as np
import torch
from torch.nn import Module


class TorchAccelerator(Module):
    def __init__(self, elements):
        Module.__init__(self)

        self.elements = {element.name: element for element in elements}

        for _, ele in self.elements.items():
            self.add_module(ele.name, ele)

    def calculate_transport(self):
        M = torch.eye(6)
        for _, ele in self.elements.items():
            M = torch.matmul(ele(), M)
        return M

    def forward(self, R):
        M = self.calculate_transport()
        return M @ R @ torch.transpose(M, 0, 1)


def rot(alpha):
    M = torch.eye(6)

    C = torch.cos(torch.tensor(alpha))
    S = torch.sin(torch.tensor(alpha))
    M[0, 0] = C
    M[0, 2] = S
    M[1, 1] = C
    M[1, 3] = S
    M[2, 0] = -S
    M[2, 2] = C
    M[3, 1] = -S
    M[3, 3] = C

    return M


class TorchQuad(Module):
    def __init__(self, name, length, K1):
        Module.__init__(self)
        self.register_parameter("L", torch.nn.parameter.Parameter(length))
        self.register_parameter("K1", torch.nn.parameter.Parameter(K1))

        self.name = name

    def forward(self):
        return self.get_matrix(self.K1)

    def get_matrix(self, K1):
        M = torch.eye(6)

        if K1 < 0.0:
            K1 = -K1
            flip = True
        elif K1 > 0.0:
            K1 = K1
            flip = False
        else:
            K1 = K1 + torch.tensor(1.0e-10)
            flip = False

        # clip to make sure we don't run into divide by zero errors
        k = torch.sqrt(K1)

        kl = self.L * k
        M[0, 0] = torch.cos(kl)
        M[0, 1] = torch.sin(kl) / k
        M[1, 0] = -k * torch.sin(kl)
        M[1, 1] = torch.cos(kl)

        M[2, 2] = torch.cosh(kl)
        M[2, 3] = torch.sinh(kl) / k
        M[3, 2] = k * torch.sinh(kl)
        M[3, 3] = torch.cosh(kl)

        if flip:
            M = rot(-np.pi / 2) @ M @ rot(np.pi / 2)

        return M


class TorchDrift(Module):
    def __init__(self, name, length, fixed=True):
        Module.__init__(self)
        self.name = name
        if fixed:
            self.register_buffer("L", torch.nn.parameter.Parameter(length))
        else:
            self.register_parameter("L", torch.nn.parameter.Parameter(length))

    def forward(self):
        M = torch.eye(6)
        M[0, 1] = self.L
        M[2, 3] = self.L

        return M
