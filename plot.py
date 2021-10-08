import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/argonne_data.txt')

fig, ax = plt.subplots(2, 1)
ax[0].plot(data.T[0])
ax[0].set_xlabel('step')
ax[0].set_ylabel('I (A)')

ax[1].plot(*data.T, '+')
ax[1].set_xlabel('I (A)')
ax[1].set_ylabel('B (T)')

fig.tight_layout()
fig.savefig('data/plot.svg')

plt.show()
