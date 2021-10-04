import matplotlib.pyplot as plt


def plot_bayes_predicition(summary, m):
    y = summary['obs']
    fig, ax = plt.subplots()
    ax.plot(m.detach(), 'C1o', label='Data')
    ax.plot(y['mean'].detach(), 'C0', label='Model prediction')
    ax.fill_between(range(len(m)),
                    y['5%'],
                    y['95%'],
                    alpha=0.25)
    ax.set_xlabel('step')
    ax.set_ylabel('B (arb. units)')
    ax.legend()

    return fig, ax
