import matplotlib.pyplot as plt
import torch

from hysteresis.states import switch

fig, ax = plt.subplots()

x = torch.linspace(-1., 1., 1000)
alpha = 0.5
beta = -alpha

c = ['C0', 'C1']
lw = 3
for idx, T in enumerate([1e-4, 5e-2]):
    pos = switch(x, alpha, T) - 1
    neg = switch(x.flipud(), beta, -T) - 1

    ax.plot(x, pos, c=c[idx], lw=lw)
    ax.plot(x.flipud(), neg, c=c[idx], lw=lw)

ax.axvline(alpha, ls='--',c='C3')
ax.axvline(beta, ls='--',c='C3')

start_idx = 500
end_idx = start_idx + 1

for ele, f in zip([x, x.flipud()], [pos, neg]):
    ax.annotate(
        '',
        xytext=(ele[start_idx], f[start_idx].detach()),
        xy=(ele[end_idx], f[end_idx].detach()),
        arrowprops=dict(arrowstyle="-|>", color='C1'),
        size=25,
        zorder=10
    )

ax.set_ylim(-1.25, 1.25)

size = torch.tensor((2, 1.5))*1.25
ax.set_ylabel('m(H)')
ax.set_xlabel('H')
fig.set_size_inches(*size)
fig.tight_layout()
fig.savefig('hysterion_loop.svg')
plt.show()
