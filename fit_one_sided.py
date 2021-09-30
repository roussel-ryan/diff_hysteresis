import hysteresis
import synthetic
import torch
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
    n_grid = 50

    h_max = 0.5
    h_min = -h_max
    b_sat = 1.0

    # get synthetic training h_data
    h, m = synthetic.generate_saturation_dataset(20, n_grid, h_max, b_sat)

    # scale m to be reasonable
    m = m / max(m)

    H = hysteresis.Hysteresis(h, h_min, h_max, b_sat, n_grid)

    # dummy predict
    m_pred = H.predict_magnetization(h).detach()

    fig, ax = plt.subplots()
    ax.plot(h, m_pred)
    ax.plot(h, m.detach(), 'o')

    # optimize
    l = train(H, m, 1000)
    m_star = H.predict_magnetization(h).detach()

    ax.plot(h, m_star)
    ax.plot()

    fig2, ax2 = plt.subplots()
    ax2.plot(l.detach())

    xx, yy = H.get_mesh()
    dens = H.get_density_matrix().detach()

    fig3, ax3 = plt.subplots()
    c = ax3.pcolor(xx, yy, dens)
    fig3.colorbar(c)

    fig4, ax4 = plt.subplots()
    c = ax4.pcolor(xx, yy, H.states[-1])
    fig4.colorbar(c)


if __name__ == '__main__':
    main()
    plt.show()
