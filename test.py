import matplotlib.pyplot as plt
import utils
import numerical_hysteresis
import synthetic

h, b = synthetic.generate_dataset()
plt.plot(h, b, 'o')
plt.show()

