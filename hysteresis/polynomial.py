import torch
from torch.nn import Module


class Polynomial(Module):
    def __init__(self, degree):
        super(Polynomial, self).__init__()
        weights = torch.zeros(degree)
        self.weights = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.p = torch.arange(1, degree + 1)

    def forward(self, x):
        xx = x.unsqueeze(-1).pow(self.p.to(x))
        return self.bias.to(x) + self.weights.to(x) @ xx.T


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from training import train_MSE

    x = torch.linspace(0, 2, 10)
    y = x ** 3 - 2.0 * x + 1.0
    poly = Polynomial(degree=3)

    #fit
    train_MSE(poly, x, y, 2000, lr=0.1)
    print(poly.bias)
    print(poly.weights)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(x, poly(x).flatten().detach())
    plt.show()