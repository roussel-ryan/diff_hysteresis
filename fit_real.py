import hysteresis
import torch
import numpy as np
import matplotlib.pyplot as plt


def loss_fn(m, m_pred):
    return torch.sum((m - m_pred) ** 2)


def train(model, m, n_steps, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)

    loss_track = []
    for i in range(n_steps):
        optimizer.zero_grad()
        output = model.predict_magnetization()
        loss = loss_fn(m, output)
        loss.backward(retain_graph=True)

        loss_track += [loss]
        optimizer.step()
        if i % 100 == 0:
            print(i)

    return torch.tensor(loss_track)


# test fitting with hysteresis class
def main():
    n_grid = 100

    h_min = 175
    h_max = 200
    b_sat = h_max

    # get real h, m
    data = torch.tensor(np.loadtxt('data/argonne_data.txt'))
    h = data.T[0]
    m = data.T[1]

    # normalize h, m
    #h = (h - min(h)) / (max(h) - min(h))
    m = (m - min(m)) / (max(m) - min(m))

    H = hysteresis.Hysteresis(h, h_min, h_max, b_sat, n_grid)

    # dummy predict
    m_pred = H.predict_magnetization(h).detach()

    fig, ax = plt.subplots()
    # ax.plot(h, m_pred)
    ax.plot(h, m.detach(), 'o', label='Data')
    ax.set_xlabel('I (A)')
    ax.set_ylabel('B (arb. units)')

    # optimize
    loss = train(H, m, 1000, lr=0.5)
    m_star = H.predict_magnetization().detach()
    ax.plot(h, m_star, label='Model Prediction')
    ax.legend()
    fig.savefig('figures/real_hysteresis_fit.svg')

    fig2, ax2 = plt.subplots()
    ax2.plot(loss.detach())

    xx, yy = H.get_mesh()
    dens = H.get_density_matrix().detach()
    raw_dens = H.get_density_matrix(raw=True).detach()

    fig3, ax3 = plt.subplots()
    c = ax3.pcolor(xx, yy, dens)
    fig3.colorbar(c, label='Hysterion Density (arb. units)')
    ax3.set_xlabel(r'$\beta$ (A)')
    ax3.set_ylabel(r'$\alpha$ (A)')

    #for ele in h:
    #    ax3.axvline(ele, c='r')
    #    ax3.axhline(ele, c='r')

    fig3.savefig('figures/real_hysteresis_density.png', dpi=300)

    fig4, ax4 = plt.subplots()
    c = ax4.pcolor(xx, yy, H.states[-1])
    fig4.colorbar(c)


if __name__ == '__main__':
    main()
    plt.show()
