import hysteresis
import synthetic
import torch
import matplotlib.pyplot as plt


def loss_fn(m, m_pred, hyst_vector):
    return torch.sum((m - m_pred) ** 2) + torch.norm(hyst_vector)**2


def train(model, m, n_steps, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)

    loss_track = []
    for i in range(n_steps):
        optimizer.zero_grad()
        output = model.predict_magnetization()
        loss = loss_fn(m, output, model.get_density_vector())
        loss.backward(retain_graph=True)

        loss_track += [loss]
        optimizer.step()
        if i % 100 == 0:
            print(i)

    return torch.tensor(loss_track)


# test fitting with hysteresis class
def main(dataset_generator):
    n_grid = 50

    h_max = 0.3
    h_min = -0.05
    b_sat = 1.0

    # get synthetic training h_data
    h, m = dataset_generator(25, n_grid*2, b_sat)

    H = hysteresis.Hysteresis(h, h_min, h_max, b_sat, n_grid)

    # dummy predict
    m_pred = H.predict_magnetization().detach()

    fig, ax = plt.subplots()
    # ax.plot(h, m_pred)
    ax.plot(h, m.detach(), 'o')

    # optimize
    l = train(H, m, 2000)
    m_star = H.predict_magnetization().detach()

    ax.plot(h, m_star)

    fig2, ax2 = plt.subplots()
    ax2.plot(l.detach())

    xx, yy = H.get_mesh()
    dens = H.get_density_matrix().detach()

    fig3, ax3 = plt.subplots()
    c = ax3.pcolor(xx, yy, dens)
    fig3.colorbar(c)

if __name__ == '__main__':
    main()
    plt.show()
